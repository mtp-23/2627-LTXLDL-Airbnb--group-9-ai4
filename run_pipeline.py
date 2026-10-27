"""Run the Santiago Airbnb data pipeline.

Pipeline order: download -> filter -> QA -> KPI -> LLM -> evaluate -> visualize.
The LLM stage is opt-in because it requires GEMINI_API_KEY and incurs API cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.download import build_jobs, download_file, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "santiago.yaml"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
REPORT_ROOT = PROJECT_ROOT / "reports"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@dataclass
class SnapshotSummary:
    snapshot: str
    listings_raw: int
    listings_filtered: int
    reviews_raw: int
    reviews_filtered: int
    duplicate_listing_ids: int
    duplicate_review_ids: int
    missing_listing_ids: int
    missing_review_comments: int
    mean_price: float | None


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)


def _source_file(snapshot: str, name: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = PROJECT_ROOT / "data" / "raw" / snapshot / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Không tìm thấy {name} của snapshot {snapshot}.")


def download_stage(config: dict[str, Any], force: bool = False) -> None:
    """Download and validate all source files using the existing downloader."""
    raw_root = PROJECT_ROOT / config["raw_dir"]
    stats = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}
    for url, destination in build_jobs(config, raw_root):
        file_type = (
            destination.name.removesuffix(".csv.gz")
            if destination.name.endswith(".csv.gz")
            else None
        )
        status = download_file(url, destination, file_type=file_type, force=force)
        stats[status] += 1
    if stats["missing"] or stats["failed"]:
        raise RuntimeError(f"Download thất bại: {stats}")
    print(f"[download] {stats}")


def filter_stage(snapshots: list[str]) -> list[SnapshotSummary]:
    """Create compact, deterministic filtered CSVs without requiring assignments."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[SnapshotSummary] = []
    for snapshot in snapshots:
        out_dir = PROCESSED_ROOT / snapshot
        out_dir.mkdir(parents=True, exist_ok=True)
        listings = _read_csv(_source_file(snapshot, "listings"))
        reviews = _read_csv(_source_file(snapshot, "reviews"))
        listings_raw_count = len(listings)
        reviews_raw_count = len(reviews)

        listing_id = _find_column(listings, "id")
        review_id = _find_column(reviews, "id")
        comments = _find_column(reviews, "comments", required=False)
        duplicate_listing_ids = int(listings[listing_id].duplicated().sum())
        duplicate_review_ids = int(reviews[review_id].duplicated().sum())
        missing_listing_ids = int(listings[listing_id].isna().sum())
        listings = listings.dropna(subset=[listing_id]).drop_duplicates(subset=[listing_id])
        reviews = reviews.dropna(subset=[review_id]).drop_duplicates(subset=[review_id])
        listings.to_csv(out_dir / "listings.csv", index=False)
        reviews.to_csv(out_dir / "reviews.csv", index=False)

        price_column = _find_column(listings, "price", required=False)
        price = (
            pd.to_numeric(
                listings[price_column]
                .astype("string")
                .str.replace(r"[^0-9,.-]", "", regex=True)
                .str.replace(",", "", regex=False),
                errors="coerce",
            )
            if price_column
            else pd.Series(dtype=float)
        )
        summaries.append(
            SnapshotSummary(
                snapshot=snapshot,
                listings_raw=listings_raw_count,
                listings_filtered=len(listings),
                reviews_raw=reviews_raw_count,
                reviews_filtered=len(reviews),
                duplicate_listing_ids=duplicate_listing_ids,
                duplicate_review_ids=duplicate_review_ids,
                missing_listing_ids=missing_listing_ids,
                missing_review_comments=int(reviews[comments].isna().sum()) if comments else 0,
                mean_price=float(price.dropna().mean()) if not price.dropna().empty else None,
            )
        )
    return summaries


def _find_column(df: pd.DataFrame, name: str, required: bool = True) -> str | None:
    target = name.casefold()
    for column in df.columns:
        if str(column).strip().lstrip("\ufeff").casefold() == target:
            return str(column)
    if required:
        raise ValueError(f"Thiếu cột '{name}'. Các cột hiện có: {list(df.columns)}")
    return None


def qa_stage(summaries: list[SnapshotSummary]) -> pd.DataFrame:
    """Persist QA checks and return one row per snapshot."""
    qa = pd.DataFrame(asdict(summary) for summary in summaries)
    qa["qa_pass"] = (
        (qa["listings_filtered"] > 0)
        & (qa["reviews_filtered"] > 0)
        & (qa["missing_listing_ids"] == 0)
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    qa.to_csv(REPORT_ROOT / "qa_report.csv", index=False)
    print(f"[qa] {int(qa['qa_pass'].sum())}/{len(qa)} snapshot đạt")
    return qa


def kpi_stage(summaries: list[SnapshotSummary]) -> pd.DataFrame:
    """Calculate operational KPIs from filtered data."""
    kpi = pd.DataFrame(
        {
            "snapshot": [s.snapshot for s in summaries],
            "listing_count": [s.listings_filtered for s in summaries],
            "review_count": [s.reviews_filtered for s in summaries],
            "review_coverage": [
                s.reviews_filtered / s.listings_filtered
                if s.listings_filtered else 0.0
                for s in summaries
            ],
            "mean_price": [s.mean_price for s in summaries],
        }
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    kpi.to_csv(REPORT_ROOT / "kpi_report.csv", index=False)
    return kpi


async def llm_stage(reviews_path: Path, output_path: Path, sample_size: int) -> None:
    from src.llm import run_llm_pipeline

    await run_llm_pipeline(
        str(reviews_path),
        str(output_path),
        sample_size=sample_size,
    )


def evaluate_stage(qa: pd.DataFrame, llm_path: Path | None) -> pd.DataFrame:
    """Evaluate stage health; LLM coverage is measured when a cache exists."""
    evaluation = qa[["snapshot", "qa_pass"]].copy()
    evaluation["llm_rows"] = 0
    if llm_path and llm_path.exists():
        llm = pd.read_json(llm_path, lines=True)
        evaluation["llm_rows"] = len(llm)
    evaluation["evaluation_pass"] = evaluation["qa_pass"]
    evaluation.to_csv(REPORT_ROOT / "evaluation.csv", index=False)
    return evaluation


def visualization_stage(kpi: pd.DataFrame) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(kpi["snapshot"], kpi["listing_count"], marker="o", label="Listings")
    axis.plot(kpi["snapshot"], kpi["review_count"], marker="o", label="Reviews")
    axis.set_title("Santiago Airbnb pipeline coverage")
    axis.set_ylabel("Rows")
    axis.tick_params(axis="x", rotation=30)
    axis.legend()
    figure.tight_layout()
    path = OUTPUT_ROOT / "pipeline_kpi.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run_pipeline(
    config_path: Path = DEFAULT_CONFIG,
    *,
    force_download: bool = False,
    skip_download: bool = False,
    run_llm: bool = False,
    sample_size: int = 3000,
) -> dict[str, Any]:
    config = load_config(config_path)
    snapshots = [str(item) for item in config["snapshots"]]
    if not skip_download:
        download_stage(config, force=force_download)
    summaries = filter_stage(snapshots)
    qa = qa_stage(summaries)
    kpi = kpi_stage(summaries)
    llm_path = OUTPUT_ROOT / "llm_results.jsonl"
    if run_llm:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("Cần đặt GEMINI_API_KEY trước khi chạy stage LLM.")
        latest_reviews = PROCESSED_ROOT / snapshots[-1] / "reviews.csv"
        asyncio.run(llm_stage(latest_reviews, llm_path, sample_size))
    evaluation = evaluate_stage(qa, llm_path if run_llm else None)
    figure = visualization_stage(kpi)
    manifest = {"qa": qa.to_dict("records"), "evaluation": evaluation.to_dict("records"), "figure": str(figure)}
    (OUTPUT_ROOT / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--llm", action="store_true", help="Chạy LLM; có thể phát sinh chi phí API.")
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size phải >= 1")
    run_pipeline(
        args.config.resolve(),
        force_download=args.force_download,
        skip_download=args.skip_download,
        run_llm=args.llm,
        sample_size=args.sample_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())