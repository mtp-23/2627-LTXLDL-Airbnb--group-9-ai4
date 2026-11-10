"""
Mô-đun trích xuất cấu trúc và phân tích cảm xúc đánh giá bằng LLM (Google Gemini).

Bao gồm:
- Tích hợp chuẩn quy trình sau QA: lọc bỏ review lỗi/trống theo cờ trong reviews_flags.csv.gz.
- Đọc cấu hình động từ configs/<city>.yaml (mốc snapshot, raw_dir, processed_dir, tham số LLM).
- Rate limiting và Retry tự động với tenacity và aiolimiter.
- Trích xuất cảm xúc (positive/mixed/negative), 7 khía cạnh (khen/chê) và ngôn ngữ.
- Cơ chế cache định dạng JSONL (hỗ trợ cờ --skip-llm để chạy offline hoặc test nhanh).
- Chuyển đổi kết quả cache sang file bảng phẳng (data/processed/llm_results.csv)
  phục vụ cho evaluate.py, kpi.py, và viz.py.
- Ước tính chi phí xử lý toàn bộ tập dữ liệu thành phố.
"""

from __future__ import annotations

import argparse
import asyncio
from enum import Enum
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
import pandas as pd
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
import yaml

load_dotenv()


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
    description: Optional[str] = Field(
        default=None,
        description="Trích dẫn ngắn gọn từ review minh chứng cho khía cạnh"
    )


class ReviewAnalysis(BaseModel):
    review_id: str
    overall_sentiment: SentimentEnum
    language: str = Field(
        default="en",
        description="Mã ngôn ngữ ISO 639-1 hai ký tự (ví dụ: en, es, vi)"
    )
    aspects: List[AspectDetail] = Field(
        default_factory=list,
        description="Danh sách các khía cạnh khen hoặc chê được đề cập"
    )


class BatchReviewResponse(BaseModel):
    results: List[ReviewAnalysis]


# ==========================================
# 2. CẤU HÌNH RATE LIMIT & RETRY
# ==========================================
# Gói Free: 15 RPM, đặt 12 RPM để an toàn tránh burst
limiter = AsyncLimiter(max_rate=12, time_period=60)


def is_rate_limit_error(exception: BaseException) -> bool:
    """Chỉ retry nếu lỗi là 429 (Too Many Requests) hoặc lỗi server 50x."""
    if isinstance(exception, APIError):
        return exception.code == 429 or exception.code >= 500
    return False


@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def process_batch_with_gemini(
    client: genai.Client,
    batch_df: pd.DataFrame,
    model_name: str = "gemini-3.5-flash-lite"
) -> List[Dict[str, Any]]:
    """Gửi 1 batch review lên Gemini và ép kiểu trả về BatchReviewResponse."""
    input_data = batch_df[["id", "comments"]].to_dict(orient="records")
    prompt = f"""Bạn là chuyên gia phân tích dữ liệu Airbnb. Hãy phân tích các đánh giá sau theo chuẩn:
1. overall_sentiment: 'positive' (chủ yếu hài lòng/khen ngợi), 'negative' (chê bai/trải nghiệm tệ), 'mixed' (cân bằng giữa khen và chê).
2. aspects: Chỉ gán các khía cạnh thuộc 7 nhóm sau:
   - 'location' (vị trí, khu vực, khoảng cách, metro)
   - 'cleanliness' (vệ sinh, sạch sẽ, bảo trì phòng)
   - 'host' (thái độ, hỗ trợ, giao tiếp của chủ nhà)
   - 'value' (đáng tiền, giá cả so với chất lượng)
   - 'amenities' (tiện nghi: bếp, wifi, máy sưởi, ban công, hồ bơi)
   - 'noise' (tiếng ồn, sự yên tĩnh, cách âm)
   - 'safety' (an toàn khu vực, khóa cửa, an ninh)
   Với mỗi khía cạnh, chỉ định rõ sentiment là 'positive' (khen) hoặc 'negative' (chê).
3. language: Mã ngôn ngữ ISO 639-1 (ví dụ: en, es, vi).
Giữ nguyên review_id từ danh sách đầu vào.

Danh sách đánh giá:
{json.dumps(input_data, ensure_ascii=False, indent=2)}
"""

    async with limiter:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchReviewResponse,
                temperature=0.0
            )
        )
        parsed = BatchReviewResponse.model_validate_json(response.text)
        return [item.model_dump() for item in parsed.results]


# ==========================================
# 3. QUẢN LÝ DỮ LIỆU SAU QA & CACHE
# ==========================================
def load_clean_reviews_after_qa(
    cfg: Dict[str, Any],
    snapshot: Optional[str] = None,
    explicit_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Nạp và làm sạch dữ liệu reviews sau khi đối chiếu với kết quả bước QA (qa.py).
    
    Quy tắc:
    1. Đọc dữ liệu reviews theo snapshot từ config.
    2. Nạp cờ QA từ data/processed/flagged/<snapshot>/reviews_flags.csv.gz.
    3. Loại bỏ các review vi phạm rule QA:
       - missing_review_text == True (comments rỗng / vô nghĩa)
       - dup_review_id_in_snapshot == True (trùng lặp review)
    4. Trả về DataFrame gồm ['id', 'comments'] sạch sẵn sàng lấy mẫu cho LLM.
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / cfg.get("raw_dir", "data/raw")
    processed_dir = project_root / cfg.get("processed_dir", "data/processed")
    snapshots = [str(s) for s in cfg.get("snapshots", [])]

    # Chọn snapshot mục tiêu (mặc định lấy snapshot mới nhất nếu không chỉ định)
    target_snapshot = snapshot or (snapshots[-1] if snapshots else None)

    reviews_file: Optional[Path] = None
    if explicit_path and Path(explicit_path).exists():
        reviews_file = Path(explicit_path)
    elif target_snapshot:
        candidate = raw_dir / target_snapshot / "reviews.csv.gz"
        if candidate.exists():
            reviews_file = candidate

    # Nếu chưa thấy snapshot chỉ định, tìm bất kỳ reviews.csv.gz trong data/raw
    if not reviews_file or not reviews_file.exists():
        candidates = list(raw_dir.glob("**/reviews.csv.gz"))
        if candidates:
            reviews_file = candidates[0]

    # Fallback kiểm thử: Nếu data/raw chưa tải, dùng tập nhãn tay để kiểm tra nhanh
    if not reviews_file or not reviews_file.exists():
        sample_file = project_root / "data" / "samples" / "hand_labeled.csv"
        if sample_file.exists():
            print(f"[QA Loader] Không thấy data/raw, nạp dữ liệu mẫu kiểm thử: {sample_file}")
            df_sample = pd.read_csv(sample_file)
            if "review_id" in df_sample.columns:
                df_sample = df_sample.rename(columns={"review_id": "id"})
            df_clean = df_sample[["id", "comments"]].dropna(subset=["comments"]).copy()
            df_clean["id"] = df_clean["id"].astype(str)
            return df_clean
        raise FileNotFoundError("Không tìm thấy file reviews trong data/raw hoặc data/samples/")

    print(f"[QA Loader] Đọc review thô từ: {reviews_file} (Snapshot: {target_snapshot})")
    df_raw = pd.read_csv(reviews_file, low_memory=False)
    if "review_id" in df_raw.columns and "id" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"review_id": "id"})

    df_raw["id"] = df_raw["id"].astype(str)
    initial_count = len(df_raw)

    # Đọc cờ QA để lọc review lỗi
    flags_file = (
        processed_dir / "flagged" / target_snapshot / "reviews_flags.csv.gz"
        if target_snapshot else None
    )
    if flags_file and flags_file.exists():
        print(f"[QA Loader] Lọc theo cờ QA từ: {flags_file}")
        df_flags = pd.read_csv(flags_file)
        df_flags["id"] = df_flags["id"].astype(str)

        # Loại bỏ các review có cờ lỗi: missing text hoặc duplicate trong snapshot
        bad_mask = pd.Series(False, index=df_flags.index)
        if "missing_review_text" in df_flags.columns:
            bad_mask |= df_flags["missing_review_text"].fillna(False).astype(bool)
        if "dup_review_id_in_snapshot" in df_flags.columns:
            bad_mask |= df_flags["dup_review_id_in_snapshot"].fillna(False).astype(bool)

        invalid_ids = set(df_flags.loc[bad_mask, "id"])
        df_clean = df_raw[~df_raw["id"].isin(invalid_ids)].copy()
        print(f"[QA Loader] QA Filter: {initial_count} review ban đầu -> loại {len(invalid_ids)} cờ xấu -> còn {len(df_clean)} review sạch.")
    else:
        print("[QA Loader] Chưa có file cờ QA, tiến hành làm sạch cơ bản (loại bỏ null/khoảng trắng)...")
        df_clean = df_raw.copy()

    df_clean = df_clean.dropna(subset=["comments"]).copy()
    df_clean = df_clean[df_clean["comments"].astype(str).str.strip() != ""]
    return df_clean[["id", "comments"]]


def load_cached_ids(cache_path: str | Path) -> Set[str]:
    """Đọc danh sách review_id đã có trong cache."""
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return set()

    processed_ids = set()
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "review_id" in data:
                    processed_ids.add(str(data["review_id"]))
            except json.JSONDecodeError:
                continue
    return processed_ids


def export_cache_to_csv(cache_path: str | Path, output_csv_path: str | Path) -> pd.DataFrame:
    """Chuyển đổi file cache JSONL sang bảng phẳng CSV tương thích với hand_labeled.csv."""
    cache_file = Path(cache_path)
    output_csv = Path(output_csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    aspects = data.get("aspects", [])
                    pos_aspects = [a["category"] for a in aspects if a.get("sentiment") == "positive"]
                    neg_aspects = [a["category"] for a in aspects if a.get("sentiment") == "negative"]

                    rows.append({
                        "review_id": str(data.get("review_id", "")),
                        "sentiment": data.get("overall_sentiment", ""),
                        "aspects_positive": ", ".join(pos_aspects),
                        "aspects_negative": ", ".join(neg_aspects),
                        "language": data.get("language", "")
                    })
                except Exception:
                    continue

    df_results = pd.DataFrame(
        rows,
        columns=["review_id", "sentiment", "aspects_positive", "aspects_negative", "language"]
    )
    df_results.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Đã xuất {len(df_results)} kết quả LLM ra: {output_csv}")
    return df_results


# ==========================================
# 4. LUỒNG CHÍNH VÀ ƯỚC TÍNH CHI PHÍ
# ==========================================
def estimate_total_cost(total_reviews: int, avg_input_chars: int = 300) -> Dict[str, Any]:
    """
    Ước tính chi phí chạy toàn bộ tập review bằng Gemini 3.5 Flash Lite.
    Giả định: 1 token ~ 4 characters.
    Giá tham khảo (Google AI Studio - Pay-as-you-go):
    - Input: $0.30 / 1 triệu token
    - Output: $2.50 / 1 triệu token
    """
    prompt_tokens = 150
    review_tokens = avg_input_chars / 4
    input_tokens_per_review = prompt_tokens + review_tokens
    output_tokens_per_review = 80

    total_input_tokens = total_reviews * input_tokens_per_review
    total_output_tokens = total_reviews * output_tokens_per_review

    input_cost = (total_input_tokens / 1_000_000) * 0.30
    output_cost = (total_output_tokens / 1_000_000) * 2.50
    total_cost = input_cost + output_cost

    return {
        "total_reviews": total_reviews,
        "estimated_input_cost_usd": round(input_cost, 4),
        "estimated_output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4)
    }


async def run_llm_pipeline(
    config_path: str = "configs/santiago.yaml",
    snapshot: Optional[str] = None,
    reviews_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    sample_size: Optional[int] = None,
    batch_size: int = 20,
    skip_llm: bool = False
) -> pd.DataFrame:
    """Chạy quy trình trích xuất LLM trên dữ liệu sau QA có cache và xuất bảng phẳng."""
    # 1. Đọc cấu hình từ YAML
    cfg: Dict[str, Any] = {}
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    llm_cfg = cfg.get("llm") or {}
    model_name = llm_cfg.get("model") or "gemini-3.5-flash-lite"
    cache_file = Path(cache_path or llm_cfg.get("cache") or "data/processed/llm_cache.jsonl")
    out_csv = Path(output_csv or "data/processed/llm_results.csv")
    target_sample_size = sample_size or llm_cfg.get("sample_size") or 1000
    seed = llm_cfg.get("seed") or 42

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Nếu bật cờ --skip-llm, chỉ đọc cache xuất ra CSV
    if skip_llm:
        print(f"[--skip-llm] Bỏ qua gọi API Gemini. Chuyển đổi cache có sẵn sang {out_csv}...")
        return export_cache_to_csv(cache_file, out_csv)

    # 2. Kiểm tra API Key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("[CẢNH BÁO] Không tìm thấy GEMINI_API_KEY hoặc API_KEY trong môi trường.")
        print("Nếu muốn chạy kiểm thử mà không gọi API, hãy sử dụng cờ: --skip-llm")
        if cache_file.exists():
            print("Đang nạp từ cache hiện có...")
            return export_cache_to_csv(cache_file, out_csv)
        raise ValueError("Thiếu API Key! Vui lòng đặt GEMINI_API_KEY trong file .env hoặc biến môi trường.")

    client = genai.Client(api_key=api_key)

    # 3. Đọc cache đã có
    processed_ids = load_cached_ids(cache_file)

    # 4. Nạp dữ liệu reviews đã được lọc sạch sau bước QA
    df_clean = load_clean_reviews_after_qa(cfg, snapshot=snapshot, explicit_path=reviews_path)

    # 5. Lấy mẫu kiểm soát hạt giống (seed)
    n_sample = min(target_sample_size, len(df_clean))
    sampled_df = df_clean.sample(n=n_sample, random_state=seed)

    # 6. Lọc bỏ các dòng đã có trong cache
    pending_df = sampled_df[~sampled_df["id"].isin(processed_ids)]
    print(f"Mẫu mục tiêu: {n_sample} | Đã có trong cache: {len(processed_ids)} | Cần gửi LLM: {len(pending_df)}")

    if not pending_df.empty:
        batches = [pending_df.iloc[i:i + batch_size] for i in range(0, len(pending_df), batch_size)]
        for i, batch_df in enumerate(batches):
            try:
                print(f"Đang xử lý batch {i+1}/{len(batches)} ({len(batch_df)} reviews)...")
                results = await process_batch_with_gemini(client, batch_df, model_name=model_name)
                with open(cache_file, "a", encoding="utf-8") as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Lỗi ở batch {i+1}: {e}")

    # 7. Xuất kết quả phẳng ra CSV cho các mô-đun kpi.py, evaluate.py, viz.py
    return export_cache_to_csv(cache_file, out_csv)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trích xuất dữ liệu có cấu trúc từ reviews sau QA bằng Gemini LLM")
    parser.add_argument("--config", default="configs/santiago.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--snapshot", default=None, help="Mốc snapshot cần phân tích (mặc định lấy mốc cuối trong config)")
    parser.add_argument("--reviews-path", default=None, help="Đường dẫn file reviews.csv(.gz) nếu chỉ định thủ công")
    parser.add_argument("--cache", default=None, help="Đường dẫn file cache JSONL")
    parser.add_argument("--output-csv", default=None, help="Đường dẫn file CSV kết quả")
    parser.add_argument("--sample-size", type=int, default=None, help="Cỡ mẫu phân tích")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua gọi API, chỉ dùng cache")
    parser.add_argument("--estimate-cost", type=int, default=None, help="Ước tính chi phí cho N reviews")

    args = parser.parse_args()

    if args.estimate_cost is not None:
        cost = estimate_total_cost(args.estimate_cost)
        print(json.dumps(cost, indent=2, ensure_ascii=False))
        return 0

    asyncio.run(
        run_llm_pipeline(
            config_path=args.config,
            snapshot=args.snapshot,
            reviews_path=args.reviews_path,
            cache_path=args.cache,
            output_csv=args.output_csv,
            sample_size=args.sample_size,
            skip_llm=args.skip_llm
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
