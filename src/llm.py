"""Mô-đun trích xuất cấu trúc và phân tích cảm xúc đánh giá bằng Gemini LLM.

- Lọc review sạch sau QA và đọc mốc từ config YAML.
- Ưu tiên 1.000 review được giao ('Mẫu LLM' == 'x' trong review.csv).
- Rate limit 12 RPM, tự động retry khi gặp lỗi 429/5xx.
- Cơ chế cache JSONL và cờ --skip-llm cho môi trường không có API Key.
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


# ==========================================
# 2. RATE LIMIT & GEMINI API CALL
# ==========================================
limiter = AsyncLimiter(max_rate=12, time_period=60)


def is_rate_limit_error(e: BaseException) -> bool:
    return isinstance(e, APIError) and (e.code == 429 or e.code >= 500)


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
    input_data = batch_df[["id", "comments"]].to_dict(orient="records")
    prompt = f"""Bạn là chuyên gia phân tích dữ liệu Airbnb. Phân tích các đánh giá sau theo chuẩn:
1. overall_sentiment: 'positive' (khen/hài lòng), 'negative' (chê/trải nghiệm tệ), 'mixed' (cân bằng khen và chê).
2. aspects: Chỉ gán các khía cạnh thuộc 7 nhóm: location, cleanliness, host, value, amenities, noise, safety.
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
                response_schema=BatchReviewResponse,
                temperature=0.0
            )
        )
        parsed = BatchReviewResponse.model_validate_json(response.text)
        return [item.model_dump() for item in parsed.results]


# ==========================================
# 3. DATA LOADER & CACHE HELPERS
# ==========================================
def get_assigned_llm_ids(project_root: Path) -> Optional[Set[str]]:
    assign_file = project_root / "data" / "samples" / "review.csv"
    if not assign_file.exists():
        return None

    df_assign = pd.read_csv(assign_file, dtype=str)
    if "Mẫu LLM" not in df_assign.columns or "review_id" not in df_assign.columns:
        return None

    is_assigned = df_assign["Mẫu LLM"].str.strip().str.lower() == "x"
    assigned = set(df_assign.loc[is_assigned, "review_id"])
    if assigned:
        print(f"[Assignment] Tìm thấy {len(assigned)} review được phân công trong review.csv")
    return assigned or None


def resolve_reviews_file(raw_dir: Path, target_snapshot: Optional[str], explicit_path: Optional[str]) -> Path:
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)
    if target_snapshot:
        candidate = raw_dir / target_snapshot / "reviews.csv.gz"
        if candidate.exists():
            return candidate

    candidates = list(raw_dir.glob("**/reviews.csv.gz"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Không tìm thấy file reviews trong {raw_dir}. Vui lòng chạy python src/download.py trước!")


def load_clean_reviews_after_qa(
    cfg: Dict[str, Any],
    snapshot: Optional[str] = None,
    explicit_path: Optional[str] = None,
    assigned_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / cfg.get("raw_dir", "data/raw")
    processed_dir = project_root / cfg.get("processed_dir", "data/processed")
    snapshots = [str(s) for s in cfg.get("snapshots", [])]
    target_snapshot = snapshot or (snapshots[-1] if snapshots else None)

    reviews_file = resolve_reviews_file(raw_dir, target_snapshot, explicit_path)
    print(f"[QA Loader] Đọc review từ: {reviews_file} (Snapshot: {target_snapshot})")

    df = pd.read_csv(reviews_file, low_memory=False)
    if "review_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"review_id": "id"})
    df["id"] = df["id"].astype(str)

    # Lọc theo cờ QA nếu có
    flags_file = processed_dir / "flagged" / target_snapshot / "reviews_flags.csv.gz" if target_snapshot else None
    if flags_file and flags_file.exists():
        flags = pd.read_csv(flags_file)
        flags["id"] = flags["id"].astype(str)
        bad_mask = pd.Series(False, index=flags.index)
        for col in ("missing_review_text", "dup_review_id_in_snapshot"):
            if col in flags.columns:
                bad_mask |= flags[col].fillna(False).astype(bool)
        invalid_ids = set(flags.loc[bad_mask, "id"])
        df = df[~df["id"].isin(invalid_ids)]
        print(f"[QA Loader] QA Filter: còn {len(df)} review sạch sau khi loại cờ xấu.")

    df = df.dropna(subset=["comments"])
    df = df[df["comments"].astype(str).str.strip().ne("")]

    if assigned_ids:
        df = df[df["id"].isin(assigned_ids)]
        print(f"[Assignment] Khớp chính xác {len(df)} / {len(assigned_ids)} review được giao.")

    return df[["id", "comments"]]


def load_cached_ids(cache_path: str | Path) -> Set[str]:
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return set()

    with open(cache_file, "r", encoding="utf-8") as f:
        return {
            str(d["review_id"])
            for line in f
            if line.strip()
            for d in [json.loads(line)]
            if "review_id" in d
        }


def export_cache_to_csv(
    cache_path: str | Path,
    output_csv_path: str | Path,
    target_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    cache_file, out_csv = Path(cache_path), Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not cache_file.exists():
        df_empty = pd.DataFrame(columns=["review_id", "sentiment", "aspects_positive", "aspects_negative", "language"])
        df_empty.to_csv(out_csv, index=False, encoding="utf-8")
        return df_empty

    rows, seen_ids = [], set()
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            rid = str(data.get("review_id", ""))
            if not rid or rid in seen_ids or (target_ids is not None and rid not in target_ids):
                continue

            seen_ids.add(rid)
            aspects = data.get("aspects", [])
            rows.append({
                "review_id": rid,
                "sentiment": data.get("overall_sentiment", ""),
                "aspects_positive": ", ".join(a["category"] for a in aspects if a.get("sentiment") == "positive"),
                "aspects_negative": ", ".join(a["category"] for a in aspects if a.get("sentiment") == "negative"),
                "language": data.get("language", "")
            })

    df = pd.DataFrame(rows, columns=["review_id", "sentiment", "aspects_positive", "aspects_negative", "language"])
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Đã xuất {len(df)} kết quả LLM ra: {out_csv}")
    return df


# ==========================================
# 4. PIPELINE & COST ESTIMATION
# ==========================================
def estimate_total_cost(total_reviews: int, avg_input_chars: int = 300) -> Dict[str, Any]:
    total_input = total_reviews * (150 + avg_input_chars / 4)
    total_output = total_reviews * 80
    input_cost = (total_input / 1_000_000) * 0.30
    output_cost = (total_output / 1_000_000) * 2.50
    return {
        "total_reviews": total_reviews,
        "estimated_input_cost_usd": round(input_cost, 4),
        "estimated_output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(input_cost + output_cost, 4)
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
    project_root = Path(__file__).resolve().parent.parent

    # Nạp cấu hình YAML
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

    assigned_ids = get_assigned_llm_ids(project_root)

    # Chế độ Skip LLM (chạy offline / test)
    if skip_llm:
        print(f"[--skip-llm] Bỏ qua gọi API Gemini. Xuất cache có sẵn sang {out_csv}...")
        return export_cache_to_csv(cache_file, out_csv, target_ids=assigned_ids)

    # Kiểm tra API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("[CẢNH BÁO] Không tìm thấy API key. Dùng --skip-llm để chạy offline bằng cache.")
        if cache_file.exists():
            return export_cache_to_csv(cache_file, out_csv, target_ids=assigned_ids)
        raise ValueError("Thiếu API Key! Đặt GEMINI_API_KEY trong .env hoặc biến môi trường.")

    client = genai.Client(api_key=api_key)
    processed_ids = load_cached_ids(cache_file)

    # Lọc dữ liệu sau QA
    df_clean = load_clean_reviews_after_qa(cfg, snapshot, reviews_path, assigned_ids)
    df_target = df_clean if assigned_ids else df_clean.sample(n=min(target_sample_size, len(df_clean)), random_state=seed)

    pending_df = df_target[~df_target["id"].isin(processed_ids)]
    print(f"Mục tiêu: {len(df_target)} | Đã cache: {len(processed_ids.intersection(set(df_target['id'])))} | Cần gửi LLM: {len(pending_df)}")

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

    return export_cache_to_csv(cache_file, out_csv, target_ids=assigned_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trích xuất dữ liệu có cấu trúc từ 1.000 review phân công bằng Gemini")
    parser.add_argument("--config", default="configs/santiago.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--snapshot", default=None, help="Mốc snapshot cần phân tích")
    parser.add_argument("--reviews-path", default=None, help="Đường dẫn file reviews.csv(.gz)")
    parser.add_argument("--cache", default=None, help="Đường dẫn file cache JSONL")
    parser.add_argument("--output-csv", default=None, help="Đường dẫn file CSV kết quả")
    parser.add_argument("--sample-size", type=int, default=None, help="Cỡ mẫu phân tích")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua gọi API, chỉ dùng cache")
    parser.add_argument("--estimate-cost", type=int, default=None, help="Ước tính chi phí cho N reviews")
    args = parser.parse_args()

    if args.estimate_cost is not None:
        print(json.dumps(estimate_total_cost(args.estimate_cost), indent=2, ensure_ascii=False))
        return 0

    asyncio.run(run_llm_pipeline(
        config_path=args.config,
        snapshot=args.snapshot,
        reviews_path=args.reviews_path,
        cache_path=args.cache,
        output_csv=args.output_csv,
        sample_size=args.sample_size,
        skip_llm=args.skip_llm
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
