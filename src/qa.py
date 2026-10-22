#!/usr/bin/env python3
"""Lọc dữ liệu Inside Airbnb theo workbook của nhóm, chạy QA và xuất KPI."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPECTED_SNAPSHOTS = 4
EXPECTED_COUNTS = {"listings": 1_000, "calendar": 365_000, "reviews": 5_000}
EXPECTED_ROOM_TYPES = {"Entire home/apt": 875, "Private room": 125}
EXPECTED_ZERO_REVIEW_COUNTS = [50, 30, 25, 25]
EXPECTED_LAST_SNAPSHOT_MISSING_PRICES = 20
EXPECTED_LLM_SAMPLE = 1_000
EXPECTED_HUMAN_LABELS = 100
EXPECTED_AREA_COUNT = 8
KEY_COLUMNS = {
    "listings": ("id",),
    "calendar": ("listing_id", "date"),
    "reviews": ("id",),
}
ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}
AVAILABILITY_VALUES = {"t", "f", "true", "false", "1", "0"}
ALIASES = {
    "id": ("id", "listing_id", "listing id", "id listing", "review_id"),
    "analysis_area": (
        "analysis_area", "analysis area", "area", "region", "zone", "khu vuc",
        "vung", "ten vung", "ma vung",
    ),
    "class": ("class", "lop", "class_name", "ten lop"),
    "group": ("group", "nhom", "group_name", "ten nhom"),
    "city": ("city", "thanh pho"),
    "llm_sample": ("mau llm", "llm sample", "sample llm", "llm"),
    "human_label": ("nhan tay", "human label", "manual label", "label"),
    "longitude_min": ("longitude min", "lon min", "min lon", "west", "kinh do tay"),
    "longitude_max": ("longitude max", "lon max", "max lon", "east", "kinh do dong"),
    "latitude_min": ("latitude min", "lat min", "min lat", "south", "vi do nam"),
    "latitude_max": ("latitude max", "lat max", "max lat", "north", "vi do bac"),
}


def normalize_name(value: Any) -> str:
    """Chuẩn hóa tên cột để nhận diện các biến thể tiếng Việt/tiếng Anh."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_column(
    frame: pd.DataFrame, semantic_name: str, required: bool = True
) -> str | None:
    """Tìm tên cột thực tế dựa trên bí danh; báo lỗi rõ nếu thiếu cột bắt buộc."""
    aliases = {normalize_name(semantic_name)}
    aliases.update(normalize_name(item) for item in ALIASES.get(semantic_name, ()))
    for column in frame.columns:
        if normalize_name(column) in aliases:
            return str(column)
    if required:
        raise ValueError(
            f"Không tìm thấy cột '{semantic_name}' (các tên nhận diện: "
            f"{', '.join(sorted(aliases))}) trong sheet/dữ liệu."
        )
    return None


def normalize_id(series: pd.Series) -> pd.Series:
    """Chuẩn hóa ID thành chuỗi để nối dữ liệu Excel và CSV ổn định."""
    values = series.astype("string").str.strip()
    return values.str.replace(r"\.0$", "", regex=True)


def is_marked(value: Any) -> bool:
    """Nhận biết dấu đánh mẫu hoặc nhãn tay trong workbook."""
    return normalize_name(value) in {"x", "yes", "true", "1", "co", "y"}


def find_workbook(project_root: Path, supplied: str | None) -> Path:
    """Tìm workbook được chỉ định hoặc một file Excel duy nhất trong configs/."""
    if supplied:
        path = Path(supplied)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy workbook Excel: {path}")
        return path

    candidates = sorted(
        list((project_root / "configs").glob("*.xlsx"))
        + list((project_root / "configs").glob("*.xlsm"))
        + list((project_root / "configs").glob("*.xls"))
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Cần chỉ định workbook bằng --mapping <file.xlsx>; "
            f"tìm thấy {len(candidates)} workbook trong configs/."
        )
    return candidates[0]


def load_group_sheet(
    workbook: Path,
    sheet_name: str,
    class_name: str | None,
    group_name: str | None,
) -> pd.DataFrame:
    """Đọc sheet và áp dụng bộ lọc Lớp/Nhóm nếu workbook có các cột này."""
    try:
        frame = pd.read_excel(workbook, sheet_name=sheet_name)
    except (ValueError, ImportError) as exc:
        raise ValueError(
            f"Không đọc được sheet '{sheet_name}' trong {workbook}: {exc}"
        ) from exc

    for semantic_name, requested in (("class", class_name), ("group", group_name)):
        column = find_column(frame, semantic_name, required=False)
        if column is None:
            if requested is not None:
                raise ValueError(
                    f"Đã truyền --{semantic_name}, nhưng sheet '{sheet_name}' "
                    f"không có cột {semantic_name}."
                )
            continue
        if requested is None:
            raise ValueError(
                f"Sheet '{sheet_name}' có cột {column}; cần truyền "
                f"--{semantic_name} để lọc đúng nhóm."
            )
        frame = frame[
            frame[column].astype("string").str.strip().str.casefold()
            == requested.strip().casefold()
        ]
    return frame.copy()


def load_mapping(
    workbook: Path, class_name: str | None, group_name: str | None, city: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc danh sách listing, review và tám vùng từ ba sheet Excel."""
    listings = load_group_sheet(workbook, "Listing", class_name, group_name)
    reviews = load_group_sheet(workbook, "Review", class_name, group_name)
    try:
        areas = pd.read_excel(workbook, sheet_name="Vùng")
    except (ValueError, ImportError) as exc:
        raise ValueError(f"Không đọc được sheet 'Vùng' trong {workbook}: {exc}") from exc
    city_col = find_column(areas, "city", required=False)
    if city_col:
        areas = areas[
            areas[city_col].astype("string").str.strip().str.casefold()
            == city.strip().casefold()
        ].copy()

    listing_id_col = find_column(listings, "id")
    listing_area_col = find_column(listings, "analysis_area", required=False)
    review_id_col = find_column(reviews, "id")
    if listing_id_col is None or review_id_col is None:
        raise ValueError("Các sheet Listing và Review phải có cột ID.")

    listings = listings.rename(columns={listing_id_col: "_id"})
    reviews = reviews.rename(columns={review_id_col: "_id"})
    listings["_id"] = normalize_id(listings["_id"])
    reviews["_id"] = normalize_id(reviews["_id"])
    if listings["_id"].duplicated().any():
        raise ValueError("Sheet Listing có ID bị trùng sau khi lọc Lớp/Nhóm.")
    if reviews["_id"].duplicated().any():
        raise ValueError("Sheet Review có ID bị trùng sau khi lọc Lớp/Nhóm.")

    if listing_area_col:
        listings = listings.rename(columns={listing_area_col: "analysis_area"})
    area_col = find_column(areas, "analysis_area", required=False)
    if area_col:
        areas = areas.rename(columns={area_col: "analysis_area"})
    return listings, reviews, areas


def load_llm_results(path: Path | None) -> pd.DataFrame:
    """Đọc kết quả LLM từ JSONL hoặc JSON; trả về review_id và sentiment."""
    if path is None:
        return pd.DataFrame(columns=["_id", "llm_sentiment"])
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy kết quả LLM: {path}")

    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL sai định dạng tại dòng {line_number}: {exc}"
                    ) from exc
                records.extend(item.get("results", [item]) if isinstance(item, dict) else [])
    else:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        records = payload.get("results", []) if isinstance(payload, dict) else payload

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["_id", "llm_sentiment"])
    id_col = find_column(frame, "id")
    sentiment_col = next(
        (
            column
            for column in frame.columns
            if normalize_name(column) in {"overall sentiment", "llm sentiment", "sentiment"}
        ),
        None,
    )
    if sentiment_col is None:
        raise ValueError(f"Không thấy cột overall_sentiment trong {path}")
    return pd.DataFrame(
        {
            "_id": normalize_id(frame[id_col]),
            "llm_sentiment": frame[sentiment_col].astype("string").str.lower(),
        }
    ).drop_duplicates("_id", keep="last")


def assign_analysis_areas(
    listings: pd.DataFrame, areas: pd.DataFrame
) -> pd.DataFrame:
    """Gán vùng theo hình chữ nhật; đối chiếu với analysis_area được cung cấp trong Listing."""
    result = listings.copy()
    existing = result.get("analysis_area", pd.Series(pd.NA, index=result.index))
    existing = existing.astype("string").str.strip().replace("", pd.NA)

    area_name_col = find_column(areas, "analysis_area")
    bounds = {
        key: find_column(areas, key)
        for key in ("longitude_min", "longitude_max", "latitude_min", "latitude_max")
    }
    if len(areas) != EXPECTED_AREA_COUNT:
        raise ValueError(
            f"Sheet 'Vùng' có {len(areas)} vùng, "
            f"yêu cầu đúng {EXPECTED_AREA_COUNT} hình chữ nhật."
        )
    if "longitude" not in result or "latitude" not in result:
        raise ValueError("Listings cần có longitude và latitude để áp dụng vùng.")

    spatial_area = pd.Series(pd.NA, index=result.index, dtype="string")
    longitude = pd.to_numeric(result["longitude"], errors="coerce")
    latitude = pd.to_numeric(result["latitude"], errors="coerce")
    for _, area in areas.iterrows():
        name = str(area[area_name_col]).strip()
        lon_min = float(area[bounds["longitude_min"]])
        lon_max = float(area[bounds["longitude_max"]])
        lat_min = float(area[bounds["latitude_min"]])
        lat_max = float(area[bounds["latitude_max"]])
        inside = (
            longitude.between(lon_min, lon_max)
            & latitude.between(lat_min, lat_max)
            & spatial_area.isna()
        )
        spatial_area.loc[inside] = name
    mismatch = existing.notna() & spatial_area.notna() & existing.ne(spatial_area)
    if mismatch.any():
        print(
            f"[CẢNH BÁO] {int(mismatch.sum())} listing có analysis_area khác "
            "kết quả áp dụng các hình chữ nhật; ưu tiên vùng hình chữ nhật."
        )
    result["analysis_area"] = spatial_area.fillna(existing)
    return result


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Chuyển cột số, hỗ trợ giá có ký hiệu tiền tệ hoặc dấu phân cách."""
    values = frame[column]
    if column == "price":
        values = values.astype("string").str.replace(r"[^0-9,.\-]", "", regex=True)
        values = values.str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def read_filtered_csv(path: Path, dataset: str, allowed_ids: set[str]) -> pd.DataFrame:
    """Đọc CSV gzip theo từng chunk và chỉ giữ ID thuộc workbook để giảm bộ nhớ."""
    key = KEY_COLUMNS[dataset][0]
    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path, compression="gzip", low_memory=False, chunksize=200_000
    ):
        missing_keys = [column for column in KEY_COLUMNS[dataset] if column not in chunk]
        if missing_keys:
            raise ValueError(
                f"{path} thiếu cột khóa: {', '.join(missing_keys)}."
            )
        chunk["_id"] = normalize_id(chunk[key])
        selected_chunks.append(chunk[chunk["_id"].isin(allowed_ids)].copy())
    if not selected_chunks:
        return pd.DataFrame(columns=[*KEY_COLUMNS[dataset], "_id"])
    return pd.concat(selected_chunks, ignore_index=True)


def add_rule_flags(
    frame: pd.DataFrame,
    dataset: str,
    snapshot: str,
    report_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Gắn cờ missing, outlier, duplicate, sai kiểu và lệch danh mục mà không xóa dòng."""
    result = frame.copy()
    result["_row_id"] = np.arange(len(result))
    flags: dict[str, pd.Series] = {}

    def record(rule: str, condition: pd.Series, reason: str, action: str) -> None:
        mask = condition.fillna(False).astype(bool)
        flags[rule] = mask if rule not in flags else (flags[rule] | mask)
        report_rows.append(
            {
                "report_type": "qa_rule",
                "snapshot": snapshot,
                "dataset": dataset,
                "rule": rule,
                "condition": reason,
                "reason": reason,
                "action": action,
                "rows_affected": int(mask.sum()),
                "rows_before": len(result),
                "rows_after": len(result),
                "kpi_name": "",
                "kpi_before": np.nan,
                "kpi_after": np.nan,
            }
        )

    missing_fields = {
        "listings": (
            "id", "price", "room_type", "latitude", "longitude",
            "review_scores_rating", "analysis_area",
        ),
        "calendar": ("listing_id", "date", "available"),
        "reviews": ("id", "date", "comments"),
    }[dataset]
    for column in missing_fields:
        if column in result.columns:
            record(
                f"missing_{column}",
                result[column].isna() | result[column].astype("string").str.strip().eq(""),
                f"{column} bị thiếu",
                "Gắn cờ; giữ dòng để truy vết.",
            )

    numeric_fields = {
        "listings": (
            "price", "latitude", "longitude", "accommodates", "bedrooms",
            "number_of_reviews", "review_scores_rating",
        ),
        "calendar": ("minimum_nights", "maximum_nights"),
        "reviews": (),
    }[dataset]
    for column in numeric_fields:
        if column not in result.columns:
            continue
        parsed = numeric_series(result, column)
        wrong_type = result[column].notna() & parsed.isna()
        record(
            f"wrong_type_{column}",
            wrong_type,
            f"{column} không chuyển được sang kiểu số",
            "Gắn cờ; không sửa hoặc xóa giá trị gốc.",
        )
        valid = parsed.dropna()
        if len(valid) >= 4:
            q1, q3 = valid.quantile([0.25, 0.75])
            spread = q3 - q1
            if spread > 0:
                outlier = parsed.lt(q1 - 1.5 * spread) | parsed.gt(q3 + 1.5 * spread)
                record(
                    f"outlier_{column}",
                    outlier,
                    f"{column} ngoài khoảng IQR 1.5",
                    "Gắn cờ ngoại lai; giữ dòng.",
                )

    if dataset == "listings" and "last_scraped" in result.columns:
        parsed_date = pd.to_datetime(result["last_scraped"], errors="coerce")
        record(
            "wrong_type_last_scraped",
            result["last_scraped"].notna() & parsed_date.isna(),
            "last_scraped không parse được thành ngày",
            "Gắn cờ; giữ giá trị gốc.",
        )
    if dataset == "calendar" and "date" in result.columns:
        parsed_date = pd.to_datetime(result["date"], errors="coerce")
        record(
            "wrong_type_date",
            result["date"].notna() & parsed_date.isna(),
            "date không parse được thành ngày",
            "Gắn cờ; giữ giá trị gốc.",
        )
        if "available" in result.columns:
            values = result["available"].astype("string").str.strip().str.lower()
            record(
                "mismatch_category_available",
                values.notna() & ~values.isin(AVAILABILITY_VALUES),
                "available không thuộc nhóm t/f hoặc true/false",
                "Gắn cờ danh mục lệch; giữ dòng.",
            )
    if dataset == "reviews" and "date" in result.columns:
        parsed_date = pd.to_datetime(result["date"], errors="coerce")
        record(
            "wrong_type_date",
            result["date"].notna() & parsed_date.isna(),
            "date không parse được thành ngày",
            "Gắn cờ; giữ giá trị gốc.",
        )
    if dataset == "reviews" and "llm_sentiment" in result.columns:
        valid_sentiments = {"positive", "mixed", "negative"}
        sentiment = result["llm_sentiment"].astype("string").str.lower()
        record(
            "missing_llm_sentiment",
            sentiment.isna() | sentiment.str.strip().eq(""),
            "Kết quả sentiment LLM bị thiếu",
            "Gắn cờ; không tự suy đoán kết quả.",
        )
        record(
            "mismatch_category_llm_sentiment",
            sentiment.notna() & ~sentiment.isin(valid_sentiments),
            "Sentiment LLM không thuộc positive/mixed/negative",
            "Gắn cờ danh mục lệch; giữ giá trị.",
        )
    if dataset == "listings" and "room_type" in result.columns:
        record(
            "mismatch_category_room_type",
            result["room_type"].notna() & ~result["room_type"].isin(ROOM_TYPES),
            "room_type không thuộc danh mục Inside Airbnb đã biết",
            "Gắn cờ danh mục lệch; giữ dòng.",
        )

    duplicate_columns = [column for column in KEY_COLUMNS[dataset] if column in result.columns]
    if duplicate_columns:
        record(
            "duplicate_within_snapshot",
            result.duplicated(subset=duplicate_columns, keep=False),
            f"trùng khóa {', '.join(duplicate_columns)} trong cùng mốc",
            "Gắn cờ tất cả bản trùng; không xóa.",
        )

    result["qa_flags"] = [
        ";".join(rule for rule, mask in flags.items() if bool(mask.iloc[index]))
        for index in range(len(result))
    ]
    return result


def check_snapshot_sanity(
    listings: pd.DataFrame,
    calendar: pd.DataFrame,
    reviews: pd.DataFrame,
    snapshot_index: int,
    snapshot: str,
) -> None:
    """So sánh số dòng, cơ cấu loại phòng, review rỗng và phân bổ vùng với yêu cầu."""
    for dataset, frame in (
        ("listings", listings),
        ("calendar", calendar),
        ("reviews", reviews),
    ):
        if len(frame) != EXPECTED_COUNTS[dataset]:
            print(
                f"[CẢNH BÁO {snapshot}] {dataset}: có {len(frame):,} dòng, "
                f"mong đợi {EXPECTED_COUNTS[dataset]:,}; kiểm tra ID và sheet Excel."
            )

    actual_rooms = listings["room_type"].value_counts(dropna=False)
    for room_type, expected in EXPECTED_ROOM_TYPES.items():
        actual = int(actual_rooms.get(room_type, 0))
        if actual != expected:
            print(
                f"[CẢNH BÁO {snapshot}] {room_type}: {actual}, "
                f"mong đợi {expected}; kiểm tra lọc ID."
            )

    if "number_of_reviews" in listings.columns:
        actual_zero = int(numeric_series(listings, "number_of_reviews").eq(0).sum())
        expected_zero = EXPECTED_ZERO_REVIEW_COUNTS[snapshot_index]
        if actual_zero != expected_zero:
            print(
                f"[CẢNH BÁO {snapshot}] listings có number_of_reviews=0: "
                f"{actual_zero}, mong đợi {expected_zero}."
            )
    if snapshot_index == EXPECTED_SNAPSHOTS - 1 and "price" in listings.columns:
        missing_prices = int(numeric_series(listings, "price").isna().sum())
        if missing_prices != EXPECTED_LAST_SNAPSHOT_MISSING_PRICES:
            print(
                f"[CẢNH BÁO {snapshot}] thiếu price: {missing_prices}, "
                f"mong đợi {EXPECTED_LAST_SNAPSHOT_MISSING_PRICES}."
            )

    if "analysis_area" in listings.columns:
        counts = listings["analysis_area"].value_counts(dropna=False)
        if listings["analysis_area"].nunique(dropna=True) != EXPECTED_AREA_COUNT:
            print(
                f"[CẢNH BÁO {snapshot}] có "
                f"{listings['analysis_area'].nunique(dropna=True)} vùng được gán, "
                f"mong đợi {EXPECTED_AREA_COUNT}."
            )
        for area, count in counts.items():
            if pd.isna(area) or not 120 <= int(count) <= 130:
                print(
                    f"[CẢNH BÁO {snapshot}] vùng {area!r} có {int(count)} listing; "
                    "mong đợi 120–130."
                )


def get_unflagged(frame: pd.DataFrame, relevant_rules: tuple[str, ...]) -> pd.Series:
    """Chọn dòng không bị gắn cờ bởi các quy tắc liên quan đến một KPI."""
    if "qa_flags" not in frame:
        return pd.Series(True, index=frame.index)
    return ~frame["qa_flags"].fillna("").str.split(";").apply(
        lambda items: any(rule in relevant_rules for rule in items)
    )


def metric_record(
    snapshot: str, name: str, target: str, formula: str, unit: str, before: Any, after: Any
) -> dict[str, Any]:
    """Chuẩn hóa một chỉ số KPI cùng mục tiêu, công thức và giá trị trước/sau QA."""
    return {
        "snapshot": snapshot,
        "kpi_name": name,
        "target": target,
        "formula": formula,
        "unit": unit,
        "value_before_qa": before,
        "value_after_qa": after,
    }


def calculate_kpis(
    listings: pd.DataFrame,
    calendar: pd.DataFrame,
    llm_reviews: pd.DataFrame,
    snapshot: str,
    currency: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Tính bảy KPI sau QA, đồng thời lưu giá trị trước QA để đối chiếu."""
    summary: list[dict[str, Any]] = []
    regional: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []

    price = numeric_series(listings, "price") if "price" in listings else pd.Series(np.nan, index=listings.index)
    valid_price = price.notna() & price.gt(0)
    clean_price = get_unflagged(
        listings, ("missing_price", "wrong_type_price", "outlier_price")
    ) & valid_price
    summary.append(
        metric_record(
            snapshot, "average_nightly_price", "Theo dõi mặt bằng giá",
            "mean(price)", f"{currency}/đêm", price[valid_price].mean(), price[clean_price].mean(),
        )
    )

    room = listings.get("room_type", pd.Series(pd.NA, index=listings.index))
    entire = room.eq("Entire home/apt")
    valid_room = room.isin(ROOM_TYPES)
    clean_room = get_unflagged(
        listings, ("missing_room_type", "mismatch_category_room_type")
    ) & valid_room
    summary.append(
        metric_record(
            snapshot, "entire_home_share", "Đo tỷ trọng căn nguyên căn",
            "count(Entire home/apt) / count(valid room_type) × 100",
            "%", entire[valid_room].mean() * 100 if valid_room.any() else np.nan,
            entire[clean_room].mean() * 100 if clean_room.any() else np.nan,
        )
    )

    if "review_scores_rating" in listings:
        rating = pd.to_numeric(listings["review_scores_rating"], errors="coerce")
    else:
        rating = pd.Series(np.nan, index=listings.index)
    valid_rating = rating.between(0, 5)
    clean_rating = valid_rating & ~listings.get(
        "qa_flags", pd.Series("", index=listings.index)
    ).str.contains(r"missing_review_scores_rating|wrong_type_review_scores_rating|outlier_review_scores_rating")
    summary.append(
        metric_record(
            snapshot, "average_rating", "Theo dõi chất lượng đánh giá",
            "mean(review_scores_rating)", "điểm / 5",
            rating[valid_rating].mean(), rating[clean_rating].mean(),
        )
    )

    areas = listings.get("analysis_area", pd.Series("unassigned", index=listings.index))
    for area in sorted(areas.dropna().astype(str).unique()):
        area_mask = areas.astype("string").eq(area)
        before = price[area_mask & valid_price].mean()
        after = price[area_mask & clean_price].mean()
        regional.append(
            {
                **metric_record(
                    snapshot, "average_nightly_price_by_area",
                    "So sánh giá giữa 8 vùng phân tích",
                    "mean(price) theo analysis_area", f"{currency}/đêm", before, after,
                ),
                "analysis_area": area,
                "listing_count": int(area_mask.sum()),
            }
        )
    unassigned = int(areas.isna().sum())
    if unassigned:
        regional.append(
            {
                **metric_record(
                    snapshot, "average_nightly_price_by_area",
                    "So sánh giá giữa 8 vùng phân tích",
                    "mean(price) theo analysis_area", f"{currency}/đêm",
                    price[areas.isna() & valid_price].mean(),
                    price[areas.isna() & clean_price].mean(),
                ),
                "analysis_area": "unassigned",
                "listing_count": unassigned,
            }
        )

    if "available" in calendar:
        available = calendar["available"].astype("string").str.lower()
        valid_availability = available.isin(AVAILABILITY_VALUES)
        unavailable = available.isin({"f", "false", "0"})
        clean_calendar = get_unflagged(
            calendar, ("missing_available", "mismatch_category_available")
        ) & valid_availability
        summary.append(
            metric_record(
                snapshot, "unavailable_day_share",
                "Ước lượng tỷ lệ ngày không khả dụng; không đồng nghĩa ngày đã đặt",
                "count(available=f) / count(valid available) × 100",
                "%", unavailable[valid_availability].mean() * 100
                if valid_availability.any() else np.nan,
                unavailable[clean_calendar].mean() * 100
                if clean_calendar.any() else np.nan,
            )
        )

    llm_valid = llm_reviews["llm_sentiment"].isin({"positive", "mixed", "negative"})
    positive = llm_reviews["llm_sentiment"].eq("positive")
    llm_clean = get_unflagged(
        llm_reviews, ("missing_llm_sentiment", "mismatch_category_llm_sentiment")
    ) & llm_valid
    sentiment_before = positive[llm_valid].mean() * 100 if llm_valid.any() else np.nan
    sentiment_after = positive[llm_clean].mean() * 100 if llm_clean.any() else np.nan
    llm_rows.append(
        {
            **metric_record(
                snapshot, "llm_positive_sentiment_share",
                "Đo tỷ lệ review tích cực trong mẫu LLM",
                "count(overall_sentiment=positive) / count(valid LLM sentiment) × 100",
                "% review", sentiment_before, sentiment_after,
            ),
            "llm_reviews": int(llm_clean.sum()),
        }
    )
    summary.append(llm_rows[-1])
    return summary, regional, llm_rows


def append_cross_snapshot_report(
    prior_ids: set[str],
    current_ids: pd.Series,
    dataset: str,
    snapshot: str,
    report_rows: list[dict[str, Any]],
) -> None:
    """Ghi nhận ID lặp giữa các mốc; giữ nguyên vì dữ liệu snapshot là ảnh chụp độc lập."""
    repeated = set(current_ids.dropna().astype(str)) & prior_ids
    report_rows.append(
        {
            "report_type": "cross_snapshot_duplicate",
            "snapshot": snapshot,
            "dataset": dataset,
            "rule": "duplicate_between_snapshots",
            "condition": "ID xuất hiện ở mốc hiện tại và ít nhất một mốc trước",
            "reason": "Bản ghi được chụp lại theo thời gian; không mặc định là lỗi.",
            "action": "Báo cáo số ID lặp, không xóa và không gắn cờ dữ liệu.",
            "rows_affected": len(repeated),
            "rows_before": len(current_ids),
            "rows_after": len(current_ids),
            "kpi_name": "",
            "kpi_before": np.nan,
            "kpi_after": np.nan,
        }
    )
    prior_ids.update(current_ids.dropna().astype(str))


def run_pipeline(args: argparse.Namespace) -> int:
    """Đọc đầu vào, lọc theo workbook, chạy QA, kiểm tra sanity và xuất dữ liệu/KPI."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else project_root / "configs" / "santiago.yaml"
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    city_slug = str(config["data_root"]).strip("/").split("/")[-1]
    workbook = find_workbook(project_root, args.mapping)
    listing_map, review_map, area_map = load_mapping(
        workbook, args.class_name, args.group_name, str(config.get("city", city_slug))
    )
    llm_path = Path(args.llm_results) if args.llm_results else None
    if llm_path is not None and not llm_path.is_absolute():
        llm_path = project_root / llm_path
    llm_results = load_llm_results(llm_path)
    if llm_results.empty:
        raise ValueError(
            "Không có kết quả LLM hợp lệ; truyền file JSONL/JSON có kết quả để tính KPI LLM."
        )

    raw_root = project_root / config.get("raw_dir", "data/raw")
    snapshots = [str(value) for value in config["snapshots"]]
    if len(snapshots) != EXPECTED_SNAPSHOTS or len(set(snapshots)) != EXPECTED_SNAPSHOTS:
        raise ValueError(f"Cần đúng {EXPECTED_SNAPSHOTS} mốc khác nhau trong config.")

    listing_ids = set(listing_map["_id"].dropna())
    review_ids = set(review_map["_id"].dropna())
    if len(listing_ids) != EXPECTED_COUNTS["listings"]:
        raise ValueError(
            f"Workbook có {len(listing_ids):,} listing ID duy nhất, mong đợi 1,000."
        )
    if len(review_ids) != EXPECTED_COUNTS["reviews"]:
        raise ValueError(
            f"Workbook có {len(review_ids):,} review ID duy nhất, mong đợi 5,000."
        )

    llm_mark_col = find_column(review_map, "llm_sample", required=False)
    human_label_col = find_column(review_map, "human_label", required=False)
    if llm_mark_col is None or human_label_col is None:
        raise ValueError("Sheet Review cần có cột 'Mẫu LLM' và 'Nhãn tay'.")
    llm_count = int(review_map[llm_mark_col].map(is_marked).sum())
    if llm_count != EXPECTED_LLM_SAMPLE:
        raise ValueError(
            f"Workbook đánh dấu {llm_count} mẫu LLM, mong đợi đúng 1,000."
        )
    human_count = int(
        review_map[human_label_col].notna()
        .astype(bool)
        .where(review_map[human_label_col].astype("string").str.strip().ne(""), False)
        .sum()
    )
    if human_count != EXPECTED_HUMAN_LABELS:
        raise ValueError(
            f"Workbook có {human_count} nhãn tay, mong đợi đúng 100."
        )

    processed_root = project_root / "data" / "processed"
    report_rows: list[dict[str, Any]] = []
    kpi_summary: list[dict[str, Any]] = []
    kpi_regions: list[dict[str, Any]] = []
    kpi_llm: list[dict[str, Any]] = []
    prior_ids = {name: set() for name in ("listings", "calendar", "reviews")}

    for index, snapshot in enumerate(snapshots):
        input_dir = raw_root / city_slug / snapshot
        file_paths = {
            name: input_dir / f"{name}.csv.gz"
            for name in ("listings", "calendar", "reviews")
        }
        missing_files = [str(path) for path in file_paths.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Thiếu dữ liệu raw ở mốc {snapshot}: {', '.join(missing_files)}"
            )

        raw = {
            "listings": read_filtered_csv(file_paths["listings"], "listings", listing_ids),
            "calendar": read_filtered_csv(file_paths["calendar"], "calendar", listing_ids),
            "reviews": read_filtered_csv(file_paths["reviews"], "reviews", review_ids),
        }

        listings = raw["listings"]
        listings = listings.merge(
            listing_map.drop_duplicates("_id"), on="_id", how="left", suffixes=("", "_mapping")
        )
        listings = assign_analysis_areas(listings, area_map)

        calendar = raw["calendar"]

        reviews = raw["reviews"]
        review_metadata = review_map.drop_duplicates("_id").copy()
        metadata_columns = ["_id"]
        for semantic in ("llm_sample", "human_label"):
            col = find_column(review_metadata, semantic, required=False)
            if col:
                review_metadata = review_metadata.rename(columns={col: semantic})
                metadata_columns.append(semantic)
        reviews = reviews.merge(
            review_metadata[metadata_columns], on="_id", how="left", suffixes=("", "_mapping")
        )
        reviews["is_llm_sample"] = (
            reviews["llm_sample"].map(is_marked)
            if "llm_sample" in reviews else False
        )
        if "human_label" in reviews:
            reviews["has_human_label"] = reviews["human_label"].map(
                lambda value: pd.notna(value) and str(value).strip() != ""
            )
        else:
            reviews["has_human_label"] = False

        if not llm_results.empty:
            reviews = reviews.merge(llm_results, on="_id", how="left")
        else:
            reviews["llm_sentiment"] = pd.NA
        check_snapshot_sanity(listings, calendar, reviews, index, snapshot)
        processed_dir = processed_root / city_slug / snapshot
        processed_dir.mkdir(parents=True, exist_ok=True)
        qa_frames = {}
        for name, frame in (
            ("listings", listings),
            ("calendar", calendar),
            ("reviews", reviews),
        ):
            append_cross_snapshot_report(
                prior_ids[name], frame["_id"], name, snapshot, report_rows
            )
            qa_frames[name] = add_rule_flags(frame, name, snapshot, report_rows)
            qa_frames[name].drop(columns=["_row_id"], errors="ignore").to_csv(
                processed_dir / f"{name}.csv", index=False
            )

        llm_for_snapshot = qa_frames["reviews"].loc[
            qa_frames["reviews"]["is_llm_sample"],
            ["_id", "llm_sentiment", "qa_flags"],
        ].drop_duplicates("_id")
        if len(llm_for_snapshot) != EXPECTED_LLM_SAMPLE:
            print(
                f"[CẢNH BÁO {snapshot}] Có {len(llm_for_snapshot)} review mẫu LLM "
                f"trên {EXPECTED_LLM_SAMPLE} được đánh dấu."
            )
        summary, regional, llm_kpi = calculate_kpis(
            qa_frames["listings"],
            qa_frames["calendar"],
            llm_for_snapshot,
            snapshot,
            str(config.get("currency", "currency configured")),
        )
        kpi_summary.extend(summary)
        kpi_regions.extend(regional)
        kpi_llm.extend(llm_kpi)
        for kpi in summary:
            report_rows.append(
                {
                    "report_type": "kpi",
                    "snapshot": snapshot,
                    "dataset": "",
                    "rule": "",
                    "condition": "",
                    "reason": "",
                    "action": "So sánh KPI trước QA và sau QA; không xóa dòng.",
                    "rows_affected": np.nan,
                    "rows_before": np.nan,
                    "rows_after": np.nan,
                    "kpi_name": kpi["kpi_name"],
                    "kpi_before": kpi["value_before_qa"],
                    "kpi_after": kpi["value_after_qa"],
                }
            )
        print(
            f"[{snapshot}] listings={len(listings):,}, calendar={len(calendar):,}, "
            f"reviews={len(reviews):,}; đã lưu vào {processed_dir}"
        )

    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_dir / "qa_report.csv", index=False)
    pd.DataFrame(kpi_summary).to_csv(processed_root / "kpi_snapshot.csv", index=False)
    pd.DataFrame(kpi_regions).to_csv(processed_root / "kpi_neighbourhood.csv", index=False)
    pd.DataFrame(kpi_llm).to_csv(processed_root / "kpi_llm.csv", index=False)
    print(f"Báo cáo QA: {report_dir / 'qa_report.csv'}")
    print(f"KPI: {processed_root / 'kpi_snapshot.csv'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Khai báo các tham số CLI cho config thành phố, workbook và kết quả LLM."""
    parser = argparse.ArgumentParser(
        description="Lọc và kiểm tra dữ liệu Inside Airbnb theo workbook của nhóm."
    )
    parser.add_argument("--config", help="YAML thành phố (mặc định: configs/santiago.yaml)")
    parser.add_argument("--mapping", help="Workbook Excel có các sheet Listing, Review, Vùng")
    parser.add_argument("--class-name", help="Giá trị lớp dùng lọc workbook nếu có cột Lớp")
    parser.add_argument("--group-name", help="Giá trị nhóm dùng lọc workbook nếu có cột Nhóm")
    parser.add_argument(
        "--llm-results", required=True, help="Kết quả LLM dạng JSONL hoặc JSON"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Chạy QA và KPI, biến lỗi đầu vào thành thông báo dễ xử lý."""
    try:
        return run_pipeline(parse_args(argv))
    except (FileNotFoundError, ValueError, KeyError, ImportError, OSError) as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())