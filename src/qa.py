import pandas as pd
import argparse
import numpy as np
from pathlib import Path
import os 
import gzip
import requests
import shutil
import yaml

# rules: name - condition - reason - todo (ưu tiên gắn cờ)
RULES = [
    # thiếu
    {
        "name": "missing_neigh_cleansed",
        "condition": "neighbourhood_cleansed bị thiếu/None/null",
        "reason": "Không thể phân nhóm listing theo khu vực",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_room_type",
        "condition": "room_type bị thiếu/None/null",
        "reason": "Không thể xác định loại phòng để tính tỷ lệ",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_number_of_reviews",
        "condition": "number_of_reviews bị thiếu/None/null",
        "reason": "Không thể xác định listing có review hay không hoặc tính số review trung bình",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_price",
        "condition": "price bị thiếu/None/null",
        "reason": "Không thể sử dụng giá listing trong các phân tích cần đến giá",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_review_date",
        "condition": "date bị thiếu/None/null trong reviews",
        "reason": "Không thể xác định thời điểm review được tạo",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_review_text",
        "condition": "comments bị thiếu/None/null hoặc chỉ chứa khoảng trắng",
        "reason": "Không có nội dung để phân tích cảm xúc bằng LLM",
        "todo": "Gắn cờ và loại khỏi tập đầu vào LLM nếu không có nội dung",
    },

    # kiểu sai
    {
        "name": "invalid_reviews_num",
        "condition": "number_of_reviews không phải số nguyên hợp lệ",
        "reason": "Không thể tính chính xác số review và các KPI liên quan",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_room_type",
        "condition": "room_type không thuộc danh mục loại phòng được chấp nhận",
        "reason": "Có thể làm sai tỷ lệ các loại phòng",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_price",
        "condition": "price không thể chuyển thành giá trị số hợp lệ",
        "reason": "Không thể sử dụng giá trong các phép tính",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_calendar_date",
        "condition": "date không chuyển được sang kiểu ngày hợp lệ",
        "reason": "Không thể phân tích lịch theo thời gian",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_calendar_available",
        "condition": "available không thuộc tập giá trị hợp lệ của nguồn dữ liệu, thường là t/f",
        "reason": "Không thể tính chính xác tỷ lệ ngày còn trống",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_review_date",
        "condition": "date của review không chuyển được sang kiểu ngày hợp lệ",
        "reason": "Không thể phân tích review theo thời gian",
        "todo": "Gắn cờ",
    },

    # ngoại lai
    {
        "name": "outlier_price",
        "condition": "price nằm ngoài ngưỡng IQR",
        "reason": "Giá bất thường có thể ảnh hưởng đến phân tích giá",
        "todo": "Gắn cờ",
    },
    # trùng
    {
        "name": "dup_listing_id",
        "condition": "id xuất hiện nhiều hơn một lần trong cùng snapshot listings",
        "reason": "Có thể làm tăng sai số lượng listing và ảnh hưởng các KPI",
        "todo": "Gắn cờ",
    },
    {
        "name": "dup_cal_key",
        "condition": "Cặp (listing_id, date) xuất hiện nhiều hơn một lần trong cùng snapshot calendar",
        "reason": "Một listing có thể bị đếm nhiều lần trong cùng một ngày",
        "todo": "Gắn cờ",
    },
    {
        "name": "dup_review_id_in_snapshot",
        "condition": "ID review xuất hiện nhiều hơn một lần trong cùng snapshot reviews",
        "reason": "Có thể khiến một review bị đếm nhiều lần",
        "todo": "Gắn cờ",
    },
    {
        "name": "dup_review_id_across_snapshots",
        "condition": "ID review xuất hiện ở nhiều snapshot khác nhau",
        "reason": "Review cũ thường được giữ lại ở snapshot sau; cần kiểm tra nhưng không tự loại",
        "todo": "Gắn cờ để kiểm tra",
    },
    # danh mục lệch
    {
        "name": "calendar_days_per_listing_mismatch",
        "condition": "Một listing có số ngày calendar khác 365 ngày trong snapshot",
        "reason": "Dữ liệu lịch của listing có thể chưa đầy đủ hoặc bị thừa",
        "todo": "Gắn cờ",
    },
]

SHEET_ID = "1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA"
project_root = Path(__file__).resolve().parent.parent
OUTPUT_DIR = project_root / "data" / "raw" / "phan_cong"
SHEETS = {
    "listing": 454441734,
    "review": 2096394307,
    "area": 903142397,
}

def download_sheet(name, gid):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SHEET_ID}/export?format=csv&gid={gid}"
    )
    output_path = OUTPUT_DIR / f"{name}.csv.gz"
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with gzip.open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)

def phan_cong():
    dataset_id = "DSE2049-C04-G09-santiago"
    config_path = project_root / "configs" / "santiago.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assignment_listing = pd.read_csv(
        OUTPUT_DIR / "listing.csv.gz", dtype=str, keep_default_na=False
    )
    assignment_review = pd.read_csv(
        OUTPUT_DIR / "review.csv.gz", dtype=str, keep_default_na=False
    )
    snapshots = config["snapshots"]
    if len(snapshots) != 4:
        raise ValueError("configs/*.yaml phải có đúng 4 snapshot")

    required_listing_columns = {"Mã dữ liệu", "listing_id"}
    required_review_columns = {"Mã dữ liệu", "review_id"}
    missing_listing_columns = required_listing_columns - set(assignment_listing.columns)
    missing_review_columns = required_review_columns - set(assignment_review.columns)
    if missing_listing_columns or missing_review_columns:
        raise ValueError(
            "Sheet phân công thiếu cột: "
            f"listing={sorted(missing_listing_columns)}, "
            f"review={sorted(missing_review_columns)}"
        )

    assigned_listing_ids = set(
        assignment_listing.loc[
            assignment_listing["Mã dữ liệu"].str.strip().eq(dataset_id), "listing_id"
        ]
    )
    assigned_review_ids = set(
        assignment_review.loc[
            assignment_review["Mã dữ liệu"].str.strip().eq(dataset_id), "review_id"
        ]
    )
    if not assigned_listing_ids or not assigned_review_ids:
        raise ValueError(f"Không tìm thấy phân công cho mã dữ liệu {dataset_id}")

    source_names = {
        "listings": ("listings.csv.gz", "id", assigned_listing_ids),
        "calendar": ("calendar.csv.gz", "listing_id", assigned_listing_ids),
        "reviews": ("reviews.csv.gz", "id", assigned_review_ids),
    }
    for snapshot in snapshots:
        source_dir = project_root / config["raw_dir"] / snapshot
        output_dir = OUTPUT_DIR / snapshot
        output_dir.mkdir(parents=True, exist_ok=True)

        for source_name, key, assigned_ids in source_names.values():
            source_path = source_dir / source_name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
            if key not in source.columns:
                raise ValueError(f"{source_path} thiếu cột khóa {key}")
            filtered = source[source[key].isin(assigned_ids)]
            filtered.to_csv(
                output_dir / source_name, index=False, compression="gzip"
            )

FLAGGED_DIR = project_root / "data" / "processed" / "flagged"
TABLE_FILES = {
    "listings": ("listings.csv.gz", ["id"]),
    "calendar": ("calendar.csv.gz", ["listing_id", "date"]),
    "reviews": ("reviews.csv.gz", ["id"]),
}

def _is_missing(series):
    values = series.astype("string").str.strip().str.lower()
    return series.isna() | values.isin(["", "none", "null", "nan"])

def _read_config():
    with (project_root / "configs" / "santiago.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)

def _read_inputs(config):
    inputs = {}
    for snapshot in config["snapshots"]:
        inputs[snapshot] = {}
        for table, (filename, keys) in TABLE_FILES.items():
            path = OUTPUT_DIR / snapshot / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            data = pd.read_csv(path, dtype=str, keep_default_na=False)
            missing_keys = set(keys) - set(data.columns)
            if missing_keys:
                raise ValueError(f"{path} thiếu cột khóa {sorted(missing_keys)}")
            inputs[snapshot][table] = data
    return inputs

def _new_flags(data, keys):
    flags = data[keys].copy()
    for key in keys:
        flags[key] = flags[key].astype(str)
    return flags

def _add_flag(flags, name, values):
    flags[name] = pd.Series(values, index=flags.index, dtype=bool)

def _numeric_price(series):
    cleaned = series.astype("string").str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")

def _build_flag_frames():
    config = _read_config()
    inputs = _read_inputs(config)
    frames = {
        snapshot: {
            table: _new_flags(data, TABLE_FILES[table][1])
            for table, data in tables.items()
        }
        for snapshot, tables in inputs.items()
    }

    review_snapshot_count = {}
    for tables in inputs.values():
        review_ids = tables["reviews"]["id"]
        for review_id in review_ids.drop_duplicates():
            review_snapshot_count[review_id] = review_snapshot_count.get(review_id, 0) + 1

    for snapshot, tables in inputs.items():
        listings = tables["listings"]
        calendar = tables["calendar"]
        reviews = tables["reviews"]
        listing_flags = frames[snapshot]["listings"]
        calendar_flags = frames[snapshot]["calendar"]
        review_flags = frames[snapshot]["reviews"]

        # Missing values.
        for rule_name, column in {
            "missing_neigh_cleansed": "neighbourhood_cleansed",
            "missing_room_type": "room_type",
            "missing_number_of_reviews": "number_of_reviews",
            "missing_price": "price",
        }.items():
            _add_flag(listing_flags, rule_name, _is_missing(listings[column]))
        _add_flag(review_flags, "missing_review_date", _is_missing(reviews["date"]))
        _add_flag(review_flags, "missing_review_text", _is_missing(reviews["comments"]))

        # Invalid types and categories.
        review_count = listings["number_of_reviews"].astype("string").str.strip()
        _add_flag(
            listing_flags,
            "invalid_reviews_num",
            review_count.ne("") & ~review_count.str.fullmatch(r"\d+", na=False),
        )
        accepted_room_types = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}
        _add_flag(
            listing_flags,
            "invalid_room_type",
            ~_is_missing(listings["room_type"]) & ~listings["room_type"].isin(accepted_room_types),
        )
        price = _numeric_price(listings["price"])
        _add_flag(listing_flags, "invalid_price", ~_is_missing(listings["price"]) & price.isna())
        _add_flag(
            calendar_flags,
            "invalid_calendar_date",
            ~_is_missing(calendar["date"]) & pd.to_datetime(calendar["date"], errors="coerce").isna(),
        )
        _add_flag(
            calendar_flags,
            "invalid_calendar_available",
            ~_is_missing(calendar["available"]) & ~calendar["available"].str.lower().isin({"t", "f"}),
        )
        _add_flag(
            review_flags,
            "invalid_review_date",
            ~_is_missing(reviews["date"]) & pd.to_datetime(reviews["date"], errors="coerce").isna(),
        )

        # Price outliers by snapshot using the IQR rule.
        valid_price = price.dropna()
        if valid_price.empty:
            outlier_price = pd.Series(False, index=listings.index)
        else:
            q1, q3 = valid_price.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_price = (price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)
        _add_flag(listing_flags, "outlier_price", outlier_price.fillna(False))

        # Duplicates inside a snapshot.
        _add_flag(listing_flags, "dup_listing_id", listings["id"].duplicated(keep=False))
        _add_flag(
            calendar_flags,
            "dup_cal_key",
            calendar.duplicated(subset=["listing_id", "date"], keep=False),
        )
        _add_flag(review_flags, "dup_review_id_in_snapshot", reviews["id"].duplicated(keep=False))
        _add_flag(
            review_flags,
            "dup_review_id_across_snapshots",
            reviews["id"].map(review_snapshot_count).fillna(0).gt(1),
        )

        # Calendar completeness: 365 rows per listing in each snapshot.
        days = calendar.groupby("listing_id").size()
        mismatch = calendar["listing_id"].map(days).ne(365)
        _add_flag(calendar_flags, "calendar_days_per_listing_mismatch", mismatch)

    return frames

def _write_flag_frames(frames, selected=None):
    selected = set(selected) if selected else None
    for snapshot, tables in frames.items():
        output_dir = FLAGGED_DIR / snapshot
        output_dir.mkdir(parents=True, exist_ok=True)
        for table, flags in tables.items():
            path = output_dir / f"{table}_flags.csv.gz"
            columns = TABLE_FILES[table][1] + [
                column for column in flags.columns
                if column not in TABLE_FILES[table][1]
                and (selected is None or column in selected)
            ]
            output = flags[columns]
            if selected and path.is_file():
                previous = pd.read_csv(path, dtype=str, keep_default_na=False)
                output = previous.merge(output, on=TABLE_FILES[table][1], how="outer")
            output.to_csv(path, index=False, compression="gzip")

def _write_qa_report(frames):
    report = []
    for snapshot, tables in frames.items():
        for table, flags in tables.items():
            total_rows = len(flags)
            key_columns = TABLE_FILES[table][1]
            for rule in RULES:
                name = rule["name"]
                if name not in flags.columns:
                    continue
                flagged_rows = int(flags[name].sum())
                report.append(
                    {
                        "snapshot": snapshot,
                        "table": table,
                        "rule_name": name,
                        "flagged_rows": flagged_rows,
                        "total_rows": total_rows,
                        "flag_rate": flagged_rows / total_rows if total_rows else 0,
                    }
                )
    pd.DataFrame(report).to_csv(
        project_root / "reports" / "qa_report.csv",
        index=False,
        encoding="utf-8",
    )

def run_qa():
    frames = _build_flag_frames()
    _write_flag_frames(frames)
    _write_qa_report(frames)

def missing():
    frames = _build_flag_frames()
    _write_flag_frames(frames, {rule["name"] for rule in RULES if rule["name"].startswith("missing_")})

def invalid():
    frames = _build_flag_frames()
    _write_flag_frames(frames, {rule["name"] for rule in RULES if rule["name"].startswith("invalid_")})

def outlier():
    frames = _build_flag_frames()
    _write_flag_frames(frames, {"outlier_price"})

def duplicate():
    frames = _build_flag_frames()
    _write_flag_frames(frames, {rule["name"] for rule in RULES if rule["name"].startswith("dup_")})

def lech():
    frames = _build_flag_frames()
    _write_flag_frames(frames, {"calendar_days_per_listing_mismatch"})

def main() -> int:
    phan_cong()
    run_qa()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())