"""Trích xuất cảm xúc, khía cạnh và ngôn ngữ của review bằng Gemini.

Nhiệm vụ duy nhất: gửi review lên LLM và trả về kết quả có cấu trúc.
Không đọc/ghi tệp (cache, CSV), không đọc cấu hình hay biến môi trường,
không chọn review (việc của run_pipeline.py). Khóa API do người gọi truyền vào.
"""
from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from aiolimiter import AsyncLimiter
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, create_model
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


# ==========================================
# 1. SCHEMAS (STRUCTURED OUTPUTS)
# ==========================================
class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    MIXED = "mixed"
    NEGATIVE = "negative"


class AspectCategoryEnum(str, Enum):
    LOCATION = "location"
    CLEANLINESS = "cleanliness"
    HOST = "host"
    VALUE = "value"
    AMENITIES = "amenities"
    NOISE = "noise"
    SAFETY = "safety"


class AspectPolarityEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class AspectDetail(BaseModel):
    category: AspectCategoryEnum
    sentiment: AspectPolarityEnum
    description: Optional[str] = Field(default=None, description="Trích dẫn ngắn gọn từ review minh chứng")


class ReviewAnalysis(BaseModel):
    review_id: str
    overall_sentiment: SentimentEnum
    language: str = Field(default="en", description="Mã ngôn ngữ ISO 639-1 hai ký tự")
    aspects: List[AspectDetail] = Field(default_factory=list, description="Danh sách các khía cạnh khen/chê")


class BatchReviewResponse(BaseModel):
    results: List[ReviewAnalysis]


# Nguồn duy nhất của tập nhãn cảm xúc (thứ tự positive, mixed, negative) và khía cạnh mặc định
SENTIMENTS = [s.value for s in SentimentEnum]
DEFAULT_ASPECTS = [a.value for a in AspectCategoryEnum]
RESULT_COLUMNS = ["review_id", "sentiment", "aspects_positive", "aspects_negative", "language"]


def _batch_schema(aspects: Optional[List[str]] = None) -> Tuple[Type[BaseModel], List[str]]:
    """Trả về (Pydantic Batch Schema, danh sách khía cạnh) khớp cấu hình."""
    if aspects is None or list(aspects) == DEFAULT_ASPECTS:
        return BatchReviewResponse, list(DEFAULT_ASPECTS)
    active = [str(a).strip().lower() for a in aspects if str(a).strip()]
    if not active:
        raise ValueError("aspects không được rỗng")
    aspect_enum = Enum("DynamicAspectCategoryEnum", {f"A{i}": a for i, a in enumerate(active)}, type=str)
    aspect_detail = create_model(
        "DynamicAspectDetail",
        category=(aspect_enum, ...),
        sentiment=(AspectPolarityEnum, ...),
        description=(Optional[str], Field(default=None, description="Trích dẫn ngắn gọn từ review minh chứng")),
    )
    review_analysis = create_model(
        "DynamicReviewAnalysis",
        review_id=(str, ...),
        overall_sentiment=(SentimentEnum, ...),
        language=(str, Field(default="en", description="Mã ngôn ngữ ISO 639-1 hai ký tự")),
        aspects=(List[aspect_detail], Field(default_factory=list, description="Danh sách các khía cạnh khen/chê")),
    )
    batch_response = create_model("DynamicBatchReviewResponse", results=(List[review_analysis], ...))
    return batch_response, active

# ==========================================
# 2. RATE LIMIT & GEMINI API CALL
# ==========================================
limiter = AsyncLimiter(max_rate=12, time_period=60)


def is_rate_limit_error(e: BaseException) -> bool:
    return isinstance(e, APIError) and (e.code == 429 or e.code >= 500)


@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
async def process_batch_with_gemini(
    client: genai.Client,
    batch_df: pd.DataFrame,
    model_name: str,
    aspects: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    schema_cls, active_aspects = _batch_schema(aspects)
    input_data = batch_df[["id", "comments"]].to_dict(orient="records")
    prompt = f"""Bạn là chuyên gia phân tích dữ liệu Airbnb. Phân tích các đánh giá sau theo chuẩn:
1. overall_sentiment: 'positive' (khen/hài lòng), 'negative' (chê/trải nghiệm tệ), 'mixed' (cân bằng khen và chê).
2. aspects: Chỉ gán các khía cạnh thuộc {len(active_aspects)} nhóm: {', '.join(active_aspects)}.
   Với mỗi khía cạnh, chỉ định rõ sentiment là 'positive' (khen) hoặc 'negative' (chê).
3. language: Mã ngôn ngữ ISO 639-1 (ví dụ: en, es, vi).
Giữ nguyên review_id.

Danh sách đánh giá:
{json.dumps(input_data, ensure_ascii=False, indent=2)}
"""
    async with limiter:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=0.0,
            ),
        )
        parsed = schema_cls.model_validate_json(response.text)
        return [item.model_dump(mode="json") for item in parsed.results]


async def analyze_reviews(
    reviews: pd.DataFrame,
    api_key: str,
    model_name: str,
    batch_size: Optional[int] = 20,
    aspects: Optional[List[str]] = None,
    on_batch: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Phân tích các review (cột id, comments) theo lô; batch lỗi được bỏ qua và ghi log.

    on_batch được gọi ngay sau mỗi lô thành công để người gọi lưu tiến độ.
    """
    batch_size = batch_size 
    client = genai.Client(api_key=api_key)
    batches = [reviews.iloc[i:i + batch_size] for i in range(0, len(reviews), batch_size)]
    results: List[Dict[str, Any]] = []
    for i, batch in enumerate(batches, 1):
        try:
            log(f"Đang xử lý batch {i}/{len(batches)} ({len(batch)} reviews)...")
            batch_results = await process_batch_with_gemini(client, batch, model_name, aspects=aspects)
        except Exception as e:  # lỗi một lô không làm dừng cả pipeline
            log(f"Lỗi ở batch {i}: {e}")
            continue
        results.extend(batch_results)
        if on_batch:
            on_batch(batch_results)
    return results


# ==========================================
# 3. CHUYỂN KẾT QUẢ & ƯỚC TÍNH CHI PHÍ
# ==========================================
def flatten_results(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Kết quả LLM (một dict/review) -> bảng phẳng: review_id, sentiment, aspects_positive, aspects_negative, language."""
    rows = []
    for data in records:
        aspects = data.get("aspects", [])
        rows.append({
            "review_id": str(data.get("review_id", "")),
            "sentiment": data.get("overall_sentiment", ""),
            "aspects_positive": ", ".join(a["category"] for a in aspects if a.get("sentiment") == "positive"),
            "aspects_negative": ", ".join(a["category"] for a in aspects if a.get("sentiment") == "negative"),
            "language": data.get("language", ""),
        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def estimate_total_cost(total_reviews: int, avg_input_chars: int = 300, batch_size: int = 20) -> Dict[str, Any]:
    """Ước tính chi phí xử lý `total_reviews` review (giá Gemini Flash-Lite, USD), tính system prompt theo lô."""
    batch_size = batch_size 
    num_batches = math.ceil(total_reviews / batch_size) if total_reviews else 0
    total_input = num_batches * 150 + total_reviews * (avg_input_chars / 4 + 15)
    total_output = total_reviews * 80
    input_cost = (total_input / 1_000_000) * 0.30
    output_cost = (total_output / 1_000_000) * 2.50
    return {
        "total_reviews": total_reviews,
        "batch_size": batch_size,
        "estimated_input_tokens": round(total_input),
        "estimated_output_tokens": round(total_output),
        "estimated_input_cost_usd": round(input_cost, 4),
        "estimated_output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(input_cost + output_cost, 4),
    }
