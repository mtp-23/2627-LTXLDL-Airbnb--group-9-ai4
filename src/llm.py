import os
import json
import asyncio
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from aiolimiter import AsyncLimiter
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

# ==========================================
# 1. ĐỊNH NGHĨA SCHEMA (STRUCTURED OUTPUTS)
# ==========================================
class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    MIXED = "mixed"
    NEGATIVE = "negative"

class AspectCategoryEnum(str, Enum):
    LOCATION = "location"
    CLEANLINESS = "cleanliness"
    HOST_SERVICE = "host_service"
    VALUE = "value"
    AMENITIES = "amenities"
    ROOM_QUALITY = "room_quality"
    OTHER = "other"

class AspectDetail(BaseModel):
    category: AspectCategoryEnum
    sentiment: SentimentEnum
    description: str = Field(description="Trích dẫn ngắn gọn từ review để minh chứng")

class ReviewAnalysis(BaseModel):
    review_id: str
    overall_sentiment: SentimentEnum
    language: str = Field(description="Mã ngôn ngữ ISO 639-1, ví dụ: en, es, vi")
    aspects: List[AspectDetail]

class BatchReviewResponse(BaseModel):
    results: List[ReviewAnalysis]

# ==========================================
# 2. CẤU HÌNH RATE LIMIT & RETRY (TENACITY)
# ==========================================
# Gói Free: 15 RPM, set 12 RPM cho an toàn (tránh burst)
limiter = AsyncLimiter(max_rate=12, time_period=60)

def is_rate_limit_error(exception):
    """Chỉ retry nếu lỗi là 429 (Too Many Requests) hoặc lỗi server 50x"""
    if isinstance(exception, APIError):
        return exception.code == 429 or exception.code >= 500
    return False

@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=4, max=60), # Chờ 4s, 8s, 16s...
    stop=stop_after_attempt(5)
)
async def process_batch_with_gemini(client: genai.Client, batch_df: pd.DataFrame) -> List[dict]:
    """Gửi 1 batch review lên Gemini và ép kiểu trả về BatchReviewResponse"""
    
    # Chuẩn bị input format: [{"review_id": "...", "comments": "..."}, ...]
    input_data = batch_df[['id', 'comments']].to_dict(orient='records')
    prompt = f"""
    Bạn là một chuyên gia phân tích dữ liệu Airbnb. Hãy phân tích danh sách các đánh giá sau đây.
    Trả về định dạng JSON chính xác theo schema được yêu cầu. Đảm bảo giữ nguyên review_id.
    
    Danh sách đánh giá:
    {json.dumps(input_data, ensure_ascii=False, indent=2)}
    """

    async with limiter:
        response = await client.aio.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchReviewResponse,
                temperature=0.0 # Bắt buộc bằng 0 để LLM trả kết quả deterministic và tuân thủ format
            )
        )
        
        # Pydantic validation: parse kết quả JSON trả về
        parsed_data = BatchReviewResponse.model_validate_json(response.text)
        return [item.model_dump() for item in parsed_data.results]

# ==========================================
# 3. QUẢN LÝ LUỒNG CHÍNH VÀ CACHE
# ==========================================
async def run_llm_pipeline(reviews_path: str, cache_path: str, sample_size: int = 3000, batch_size: int = 20):
    # Khởi tạo client (Tự động đọc GEMINI_API_KEY từ biến môi trường)
    client = genai.Client()
    
    # 1. Đọc cache nếu có
    processed_ids = set()
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                processed_ids.add(json.loads(line)['review_id'])
    
    # 2. Đọc và lấy mẫu dữ liệu
    df = pd.read_csv(reviews_path, usecols=['id', 'comments'])
    df = df.dropna(subset=['comments']).copy()
    
    # Đảm bảo string
    df['id'] = df['id'].astype(str)
    
    # Sample (lấy random state cố định để tái lập được)
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    # Lọc bỏ các dòng đã có trong cache
    pending_df = sampled_df[~sampled_df['id'].isin(processed_ids)]
    print(f"Total sampled: {sample_size}. Already processed: {len(processed_ids)}. Pending: {len(pending_df)}")

    if pending_df.empty:
        return
        
    # 3. Chia batch
    batches = [pending_df.iloc[i:i + batch_size] for i in range(0, len(pending_df), batch_size)]
    
    # 4. Xử lý Async từng batch và ghi file liên tục
    for i, batch_df in enumerate(batches):
        try:
            print(f"Processing batch {i+1}/{len(batches)}...")
            results = await process_batch_with_gemini(client, batch_df)
            
            # Ghi nối (append) ngay vào cache
            with open(cache_path, 'a', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            print(f"Lỗi ở batch {i+1}: {e}")
            # Tuỳ chọn: Có thể break để dừng chương trình hoặc continue để bỏ qua batch lỗi
def estimate_total_cost(total_reviews: int, avg_input_chars: int = 300) -> dict:
    """
    Ước tính chi phí chạy toàn bộ tập review bằng Gemini 1.5 Flash.
    Giả định: 1 token ~ 4 characters.
    Giá tham khảo (Google AI Studio - Tier Pay-as-you-go):
    - Input: $0.075 / 1 triệu token
    - Output: $0.30 / 1 triệu token
    """
    # 1. Ước tính số token cho mỗi review
    prompt_tokens = 150 # System prompt + định dạng JSON
    review_tokens = avg_input_chars / 4
    input_tokens_per_review = prompt_tokens + review_tokens
    output_tokens_per_review = 80 # Trả về JSON (sentiment, aspect, language)
    
    # 2. Tính tổng token
    total_input_tokens = total_reviews * input_tokens_per_review
    total_output_tokens = total_reviews * output_tokens_per_review
    
    # 3. Tính chi phí (USD)
    input_cost = (total_input_tokens / 1_000_000) * 0.075
    output_cost = (total_output_tokens / 1_000_000) * 0.30
    total_cost = input_cost + output_cost
    
    return {
        "total_reviews": total_reviews,
        "estimated_input_cost_usd": round(input_cost, 4),
        "estimated_output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4)
    }