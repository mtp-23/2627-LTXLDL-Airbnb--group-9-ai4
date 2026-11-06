import argparse
import gzip
import io
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import yaml


GROUP_CODE = "DSE2049-C04-G09-santiago"
SHEET_NAMES = ("listing.csv", "review.csv", "area.csv")
SHEET_URLS = {
    "listing.csv": "https://docs.google.com/spreadsheets/d/1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA/edit?gid=454441734#gid=454441734",
    "review.csv": "https://docs.google.com/spreadsheets/d/1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA/edit?gid=2096394307#gid=2096394307",
    "area.csv": "https://docs.google.com/spreadsheets/d/1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA/edit?gid=903142397#gid=903142397",
}
COORDINATE_RULE = "analysis_area_coordinate_valid"
VALID_ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}


# rules: name - condition - reason - todo (ưu tiên gắn cờ)
RULES = [
    # thiếu
    {"name": "missing_neigh_cleansed", "condition": "neighbourhood_cleansed bị thiếu/None/null", "reason": "Không thể phân nhóm listing theo khu vực", "todo": "Gắn cờ"},
    {"name": "missing_room_type", "condition": "room_type bị thiếu/None/null", "reason": "Không thể xác định loại phòng để tính tỷ lệ", "todo": "Gắn cờ"},
    {"name": "missing_number_of_reviews", "condition": "number_of_reviews bị thiếu/None/null", "reason": "Không thể xác định listing có review hay không hoặc tính số review trung bình", "todo": "Gắn cờ"},
    {"name": "missing_price", "condition": "price bị thiếu/None/null", "reason": "Không thể sử dụng giá listing trong các phân tích cần đến giá", "todo": "Gắn cờ"},
    {"name": "missing_review_text", "condition": "comments bị thiếu/None/null hoặc chỉ chứa khoảng trắng", "reason": "Không có nội dung để phân tích cảm xúc bằng LLM", "todo": "Gắn cờ và loại khỏi tập đầu vào LLM nếu không có nội dung"},
    # kiểu sai
    {"name": "invalid_room_type", "condition": "room_type không thuộc danh mục loại phòng được chấp nhận", "reason": "Có thể làm sai tỷ lệ các loại phòng", "todo": "Gắn cờ"},
    {"name": "invalid_price", "condition": "price không thể chuyển thành giá trị số hợp lệ", "reason": "Không thể sử dụng giá trong các phép tính", "todo": "Gắn cờ"},
    {"name": "invalid_calendar_date", "condition": "date không chuyển được sang kiểu ngày hợp lệ", "reason": "Không thể phân tích lịch theo thời gian", "todo": "Gắn cờ"},
    {"name": "invalid_calendar_available", "condition": "available không thuộc tập giá trị hợp lệ của nguồn dữ liệu, thường là t/f", "reason": "Không thể tính chính xác tỷ lệ ngày còn trống", "todo": "Gắn cờ"},
    {"name": "invalid_review_date", "condition": "date của review không chuyển được sang kiểu ngày hợp lệ", "reason": "Không thể phân tích review theo thời gian", "todo": "Gắn cờ"},
    # ngoại lai
    {"name": "outlier_price", "condition": "price nằm ngoài ngưỡng IQR", "reason": "Giá bất thường có thể ảnh hưởng đến phân tích giá", "todo": "Gắn cờ"},
    # trùng
    {"name": "dup_list_id", "condition": "id xuất hiện nhiều hơn một lần trong cùng snapshot listings", "reason": "Có thể làm tăng sai số lượng listing và ảnh hưởng các KPI", "todo": "Gắn cờ"},
    {"name": "dup_cal_key", "condition": "Cặp (listing_id, date) xuất hiện nhiều hơn một lần trong cùng snapshot calendar", "reason": "Một listing có thể bị đếm nhiều lần trong cùng một ngày", "todo": "Gắn cờ"},
    {"name": "dup_review_id", "condition": "ID review xuất hiện nhiều hơn một lần trong cùng snapshot reviews", "reason": "Có thể khiến một review bị đếm nhiều lần", "todo": "Gắn cờ"},
    # danh mục lệch
]
RULE_NAMES = [rule["name"] for rule in RULES]


def find_project_root() -> Path:
    """Return the repository root based on this module's location."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the existing project configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config.get("snapshots"), list) or not config["snapshots"]:
        raise ValueError("Config phải chứa danh sách snapshots không rỗng")
    for key in ("city", "raw_dir", "processed_dir"):
        if key not in config:
            raise ValueError(f"Config thiếu khóa '{key}'")
    return config


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except Exception as error:
        raise ValueError(f"Không đọc được CSV {path}: {error}") from error


def _require_columns(frame: pd.DataFrame, columns: list[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} thiếu cột: {', '.join(missing)}")


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip().casefold()


def _id(value: Any) -> str:
    value = _text(value)
    return re.sub(r"\.0+$", "", value) if re.fullmatch(r"[+-]?\d+\.0+", value) else value


def _atomic_bytes(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(content)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _sheet_export_url(url: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/spreadsheets/d/([^/]+)", parsed.path)
    gid = parse_qs(parsed.query).get("gid", [None])[0]
    if gid is None and parsed.fragment.startswith("gid="):
        gid = parsed.fragment.removeprefix("gid=")
    if not match or not gid:
        raise ValueError(f"URL Google Sheet không hợp lệ: {url}")
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv&gid={gid}"


def _validate_sheet_content(content: bytes, name: str) -> None:
    if not content or content.lstrip().startswith((b"<", b"<!DOCTYPE")):
        raise ValueError(f"Google trả về HTML hoặc nội dung rỗng cho {name}")
    try:
        frame = pd.read_csv(io.BytesIO(content), nrows=5, encoding="utf-8-sig")
    except Exception as error:
        raise ValueError(f"Nội dung tải về không phải CSV cho {name}: {error}") from error
    if frame.empty and len(frame.columns) == 0:
        raise ValueError(f"Sheet {name} không có header")


def download_assignment_sheets(raw_dir: Path, urls: dict[str, str] = SHEET_URLS) -> dict[str, Path]:
    """Download and verify the three assignment tabs as real gzip CSV files."""
    destination = raw_dir / "phan_cong"
    paths: dict[str, Path] = {}
    for name in SHEET_NAMES:
        try:
            response = requests.get(_sheet_export_url(urls[name]), timeout=60)
            response.raise_for_status()
            _validate_sheet_content(response.content, name)
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f"Không tải được sheet {name}: {error}") from error
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb") as file:
            file.write(response.content)
        target = destination / f"{name}.gz"
        _atomic_bytes(target, compressed.getvalue())
        paths[name] = target
    return paths


def copy_assignment_sheets(sheet_dir: Path, raw_dir: Path) -> dict[str, Path]:
    """Copy local CSV or CSV.GZ assignment sheets into raw/phan_cong."""
    destination = raw_dir / "phan_cong"
    paths: dict[str, Path] = {}
    for name in SHEET_NAMES:
        stem = name.removesuffix(".csv")
        candidates = [
            sheet_dir / name,
            sheet_dir / f"{name}.gz",
            sheet_dir / f"{stem.title()}.csv.gz",
            sheet_dir / f"{stem.title()}.csv",
        ]
        if name == "area.csv":
            candidates.extend([sheet_dir / "Vùng.csv.gz", sheet_dir / "Vùng.csv", sheet_dir / "vung.csv.gz", sheet_dir / "vung.csv"])
        source = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not source.is_file():
            raise FileNotFoundError(f"Thiếu sheet phân công: {source}")
        target = destination / f"{name}.gz"
        if source.suffix == ".gz":
            _atomic_bytes(target, source.read_bytes())
        else:
            compressed = io.BytesIO()
            with gzip.GzipFile(fileobj=compressed, mode="wb") as file:
                file.write(source.read_bytes())
            _atomic_bytes(target, compressed.getvalue())
        paths[name] = target
    return paths


def load_assignment_sheets(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate the three downloaded assignment sheets."""
    listing = _read_csv(paths["listing.csv"], dtype=str, compression="gzip")
    review = _read_csv(paths["review.csv"], dtype=str, compression="gzip")
    area = _read_csv(paths["area.csv"], dtype=str, compression="gzip")
    _require_columns(listing, ["Mã dữ liệu", "listing_id", "analysis_area"], paths["listing.csv"])
    _require_columns(review, ["Mã dữ liệu", "review_id", "listing_id"], paths["review.csv"])
    _require_columns(area, ["Thành phố", "analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"], paths["area.csv"])
    return listing, review, area


def validate_assignments(listing: pd.DataFrame, review: pd.DataFrame, area: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate selected assignment rows and report duplicate/conflicting IDs."""
    selected_listing = listing[listing["Mã dữ liệu"].map(_text).eq(GROUP_CODE.casefold())].copy()
    selected_review = review[review["Mã dữ liệu"].map(_text).eq(GROUP_CODE.casefold())].copy()
    selected_listing["_id"] = selected_listing["listing_id"].map(_id)
    selected_review["_id"] = selected_review["review_id"].map(_id)
    rows: list[dict[str, Any]] = []

    def add(dataset: str, rule: str, observed: Any, expected: Any, status: str, details: str = "") -> None:
        rows.append({"category": "assignment", "snapshot": "", "dataset": dataset, "rule": rule, "status": status, "observed": observed, "expected": expected, "details": details})

    for frame, dataset in ((selected_listing, "listing_assignment"), (selected_review, "review_assignment")):
        add(dataset, "assigned_rows", len(frame), ">= 1", "PASS" if len(frame) else "ERROR")
        missing = int(frame["_id"].eq("").sum())
        duplicate = int(frame.loc[frame["_id"].ne("")].duplicated("_id", keep=False).sum())
        add(dataset, "missing_assignment_id", missing, 0, "PASS" if missing == 0 else "FAIL", "ID trống")
        add(dataset, "duplicate_assignment_id", duplicate, 0, "PASS" if duplicate == 0 else "FAIL", "ID lặp")
    selected_listing["_area"] = selected_listing["analysis_area"].map(_text)
    conflicts = selected_listing.groupby("_id")["_area"].nunique()
    conflict_count = int((conflicts > 1).sum())
    add("listing_assignment", "conflicting_area_mapping", conflict_count, 0, "PASS" if conflict_count == 0 else "ERROR", "Một listing có nhiều vùng")
    city_rows = int(area["Thành phố"].map(_text).eq("santiago").sum())
    add("area_assignment", "santiago_area_rows", city_rows, ">= 1", "PASS" if city_rows else "ERROR")
    return rows


def get_group_assignments(listing: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized listing-area and review-ID assignments for this group."""
    selected_listing = listing[listing["Mã dữ liệu"].map(_text).eq(GROUP_CODE.casefold())].copy()
    selected_review = review[review["Mã dữ liệu"].map(_text).eq(GROUP_CODE.casefold())].copy()
    selected_listing["_id"] = selected_listing["listing_id"].map(_id)
    selected_listing["_area"] = selected_listing["analysis_area"].map(_text)
    selected_review["_id"] = selected_review["review_id"].map(_id)
    if selected_listing["_id"].eq("").any() or selected_review["_id"].eq("").any():
        raise ValueError("Assignment có ID trống")
    if selected_listing.groupby("_id")["_area"].nunique().gt(1).any():
        raise ValueError("Assignment có listing_id được gán cho nhiều analysis_area")
    return selected_listing.drop_duplicates("_id"), selected_review.drop_duplicates("_id")


def load_region_bounds(area: pd.DataFrame, city: str) -> pd.DataFrame:
    """Load valid bounding boxes for the configured city."""
    result = area[area["Thành phố"].map(_text).eq(_text(city))].copy()
    if result.empty:
        raise ValueError(f"Không tìm thấy thành phố {city} trong sheet Vùng")
    result["_area"] = result["analysis_area"].map(_text)
    if result["_area"].eq("").any() or result["_area"].duplicated().any():
        raise ValueError("Sheet Vùng có analysis_area thiếu hoặc trùng")
    for column in ("lon_min", "lat_min", "lon_max", "lat_max"):
        result[column] = pd.to_numeric(result[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    invalid = result[["lon_min", "lat_min", "lon_max", "lat_max"]].isna().any(axis=1)
    invalid |= result["lon_min"] > result["lon_max"]
    invalid |= result["lat_min"] > result["lat_max"]
    if invalid.any():
        raise ValueError("Sheet Vùng có bounding box thiếu hoặc không hợp lệ")
    return result


def filter_snapshot(raw_dir: Path, snapshot: str, listing_assignment: pd.DataFrame, review_assignment: pd.DataFrame) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, Any]]:
    """Filter listings, calendar and reviews for one snapshot."""
    folder = raw_dir / snapshot
    listings = _read_csv(folder / "listings.csv.gz", dtype=str, compression="gzip")
    calendar = _read_csv(folder / "calendar.csv.gz", dtype=str, compression="gzip")
    reviews = _read_csv(folder / "reviews.csv.gz", dtype=str, compression="gzip")
    _require_columns(listings, ["id"], folder / "listings.csv.gz")
    _require_columns(calendar, ["listing_id"], folder / "calendar.csv.gz")
    _require_columns(reviews, ["id", "listing_id"], folder / "reviews.csv.gz")
    listing_ids = set(listing_assignment["_id"])
    listing_keys = listings["id"].map(_id)
    filtered_listings = listings.loc[listing_keys.isin(listing_ids)].copy()
    assignment_columns = listing_assignment[["_id", "analysis_area", "_area"]].drop_duplicates("_id")
    filtered_listings["_id"] = filtered_listings["id"].map(_id)
    filtered_listings = filtered_listings.merge(assignment_columns, on="_id", how="left", suffixes=("", "_assigned"))
    if "analysis_area" in listings.columns:
        raw_area = filtered_listings["analysis_area"].map(_text)
        assigned_area = filtered_listings["analysis_area_assigned"].map(_text)
        filtered_listings["_area_mapping_mismatch"] = raw_area.ne("") & raw_area.ne(assigned_area)
        filtered_listings["analysis_area"] = filtered_listings["analysis_area_assigned"]
    else:
        filtered_listings["analysis_area"] = filtered_listings["analysis_area"]
        filtered_listings["_area_mapping_mismatch"] = False
    filtered_listings = filtered_listings.drop(columns=["_id", "_area", "analysis_area_assigned"], errors="ignore")
    calendar_keys = calendar["listing_id"].map(_id)
    filtered_calendar = calendar.loc[calendar_keys.isin(listing_ids)].copy()
    review_ids = set(review_assignment["_id"])
    review_keys = reviews["id"].map(_id)
    filtered_reviews = reviews.loc[review_keys.isin(review_ids)].copy()
    stats = {
        "listings": {"raw_rows": len(listings), "filtered_rows": len(filtered_listings), "assigned_found": filtered_listings["id"].map(_id).nunique(), "assigned_missing": len(listing_ids - set(listing_keys)), "area_mapping_mismatch": int(filtered_listings["_area_mapping_mismatch"].sum())},
        "calendar": {"raw_rows": len(calendar), "filtered_rows": len(filtered_calendar), "assigned_found": filtered_calendar["listing_id"].map(_id).nunique()},
        "reviews": {"raw_rows": len(reviews), "filtered_rows": len(filtered_reviews), "assigned_found": filtered_reviews["id"].map(_id).nunique(), "assigned_missing": len(review_ids - set(review_keys))},
    }
    filtered_listings = filtered_listings.drop(columns=["_area_mapping_mismatch"], errors="ignore")
    return (filtered_listings, filtered_calendar, filtered_reviews), stats


def _missing(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(True, index=frame.index)
    return frame[column].isna() | frame[column].astype(str).str.strip().str.casefold().isin({"", "nan", "none", "null"})


def _flags(frame: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    result = frame.copy()
    for name in RULE_NAMES + (extra or []):
        result[name] = False
    return result


def apply_listing_flags(listings: pd.DataFrame) -> pd.DataFrame:
    """Apply listing row-level rules."""
    result = _flags(listings, [COORDINATE_RULE])
    result.loc[_missing(result, "neighbourhood_cleansed"), "missing_neigh_cleansed"] = True
    result.loc[_missing(result, "room_type"), "missing_room_type"] = True
    result.loc[_missing(result, "number_of_reviews"), "missing_number_of_reviews"] = True
    result.loc[_missing(result, "price"), "missing_price"] = True
    if "room_type" in result:
        valid = result["room_type"].astype(str).str.strip().isin(VALID_ROOM_TYPES)
        result.loc[~_missing(result, "room_type") & ~valid, "invalid_room_type"] = True
    if "price" in result:
        prices = pd.to_numeric(result["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        result.loc[~_missing(result, "price") & prices.isna(), "invalid_price"] = True
        valid_prices = prices.dropna()
        if len(valid_prices) >= 4:
            q1, q3 = valid_prices.quantile([0.25, 0.75])
            iqr = q3 - q1
            result.loc[prices.lt(q1 - 1.5 * iqr) | prices.gt(q3 + 1.5 * iqr), "outlier_price"] = True
    result.loc[result["id"].duplicated(keep=False), "dup_list_id"] = True
    return result


def apply_calendar_flags(calendar: pd.DataFrame) -> pd.DataFrame:
    """Apply calendar row-level rules."""
    result = _flags(calendar)
    if "date" in result:
        parsed = pd.to_datetime(result["date"], errors="coerce")
        result.loc[~_missing(result, "date") & parsed.isna(), "invalid_calendar_date"] = True
    if "available" in result:
        available = result["available"].astype(str).str.strip().str.casefold()
        result.loc[~available.isin({"t", "f"}), "invalid_calendar_available"] = True
    if {"listing_id", "date"}.issubset(result.columns):
        result.loc[result.duplicated(["listing_id", "date"], keep=False), "dup_cal_key"] = True
    return result


def apply_review_flags(reviews: pd.DataFrame) -> pd.DataFrame:
    """Apply review row-level rules."""
    result = _flags(reviews)
    result.loc[_missing(result, "comments"), "missing_review_text"] = True
    if "date" in result:
        parsed = pd.to_datetime(result["date"], errors="coerce")
        result.loc[~_missing(result, "date") & parsed.isna(), "invalid_review_date"] = True
    result.loc[result["id"].duplicated(keep=False), "dup_review_id"] = True
    return result


def check_analysis_area_coordinates(listings: pd.DataFrame, bounds: pd.DataFrame) -> dict[str, Any]:
    """Check assigned areas against all Santiago bounding boxes."""
    flags = pd.Series(False, index=listings.index, dtype=bool)
    longitude = pd.to_numeric(listings.get("longitude", pd.Series(index=listings.index)), errors="coerce")
    latitude = pd.to_numeric(listings.get("latitude", pd.Series(index=listings.index)), errors="coerce")
    assigned = listings.get("analysis_area", pd.Series(index=listings.index)).map(_text)
    report: dict[str, Any] = {"flags": flags, "total": len(listings), "valid": 0, "invalid": 0, "missing_longitude": int(longitude.isna().sum()), "missing_latitude": int(latitude.isna().sum()), "missing_area": int(assigned.eq("").sum()), "area_not_found": 0, "outside_all": 0, "overlap": 0, "assigned_mismatch": 0, "mismatch_ids": []}
    known_areas = set(bounds["_area"])
    for index in listings.index:
        if longitude.loc[index] != longitude.loc[index] or latitude.loc[index] != latitude.loc[index] or not assigned.loc[index]:
            report["invalid"] += 1
            continue
        containing = bounds.loc[(bounds["lon_min"] <= longitude.loc[index]) & (longitude.loc[index] <= bounds["lon_max"]) & (bounds["lat_min"] <= latitude.loc[index]) & (latitude.loc[index] <= bounds["lat_max"]), "_area"].tolist()
        if len(containing) > 1:
            report["overlap"] += 1
        if not containing:
            report["outside_all"] += 1
        if assigned.loc[index] not in known_areas:
            report["area_not_found"] += 1
        valid = assigned.loc[index] in containing
        flags.loc[index] = valid
        if valid:
            report["valid"] += 1
        else:
            report["invalid"] += 1
            report["assigned_mismatch"] += 1
            report["mismatch_ids"].append(str(listings.loc[index, "id"]))
    return report


def _atomic_write_frame(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".csv.gz", delete=False) as file:
        temporary = Path(file.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8", compression="gzip")
        pd.read_csv(temporary, compression="gzip", nrows=1)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def save_processed_data(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: Path, snapshot: str) -> None:
    """Save filtered data under processed/<snapshot>."""
    for frame, name in zip(data, ("listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz")):
        _atomic_write_frame(frame, output_dir / snapshot / name)


def save_flagged_data(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: Path, snapshot: str) -> None:
    """Save flagged data under processed/flagged/<snapshot>."""
    for frame, name in zip(data, ("listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz")):
        _atomic_write_frame(frame, output_dir / "flagged" / snapshot / name)


def _row_report(snapshot: str, dataset: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    rule_names = RULE_NAMES + ([COORDINATE_RULE] if dataset == "listings" else [])
    for rule in rule_names:
        true_count = int(frame[rule].sum())
        if rule == COORDINATE_RULE:
            status = "PASS" if true_count == len(frame) else "FAIL"
            observed, expected = len(frame) - true_count, 0
            details = f"invalid={len(frame) - true_count}, valid={true_count}"
        else:
            status = "PASS" if true_count == 0 else "FAIL"
            observed, expected = true_count, 0
            details = f"rows={len(frame)}, false={len(frame) - true_count}"
        rows.append({"category": "row_rule", "snapshot": snapshot, "dataset": dataset, "rule": rule, "status": status, "observed": observed, "expected": expected, "details": details})
    return rows


def run_dataset_level_qa(snapshot: str, snapshot_index: int, filtered: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], listing_assignment: pd.DataFrame, review_assignment: pd.DataFrame) -> list[dict[str, Any]]:
    """Run project-level expectations without altering data to force PASS."""
    listings, calendar, reviews = filtered
    rows: list[dict[str, Any]] = []

    def add(rule: str, observed: Any, expected: Any, status: str | None = None, details: str = "") -> None:
        rows.append({"category": "dataset", "snapshot": snapshot, "dataset": "all", "rule": rule, "status": status or ("PASS" if observed == expected else "FAIL"), "observed": observed, "expected": expected, "details": details})

    add("assigned_listing_count", listings["id"].map(_id).nunique(), len(listing_assignment))
    add("assigned_review_count", reviews["id"].map(_id).nunique(), len(review_assignment))
    add("calendar_row_count", len(calendar), 365000)
    add("analysis_area_count", listings["analysis_area"].map(_text).nunique(), 8)
    area_counts = listings["analysis_area"].map(_text).value_counts().to_dict()
    area_in_range = bool(area_counts) and all(120 <= count <= 130 for count in area_counts.values())
    add("analysis_area_distribution", area_counts, "8 areas, each 120-130", "PASS" if area_in_range else "FAIL")
    room_counts = listings.get("room_type", pd.Series(dtype=str)).value_counts().to_dict()
    add("room_type_distribution", room_counts, {"Entire home/apt": 875, "Private room": 125}, "PASS" if room_counts.get("Entire home/apt", 0) == 875 and room_counts.get("Private room", 0) == 125 else "FAIL")
    zero_reviews = int(pd.to_numeric(listings.get("number_of_reviews", pd.Series(dtype=str)), errors="coerce").eq(0).sum())
    expected_zero_reviews = (50, 30, 25, 25)[snapshot_index] if snapshot_index < 4 else "snapshot-specific expectation"
    add("zero_review_listings", zero_reviews, expected_zero_reviews, "PASS" if zero_reviews == expected_zero_reviews else "FAIL")
    missing_price = int(_missing(listings, "price").sum())
    expected_missing_price = 20 if snapshot_index == 3 else "not specified"
    add("missing_price_count", missing_price, expected_missing_price, "PASS" if snapshot_index != 3 or missing_price == expected_missing_price else "FAIL")
    return rows


def run_cross_snapshot_qa(snapshot_data: list[tuple[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]) -> list[dict[str, Any]]:
    """Compare assigned listing and review ID sets across all snapshots."""
    rows: list[dict[str, Any]] = []
    if not snapshot_data:
        return rows
    base_snapshot, base_data = snapshot_data[0]
    base_listing_ids = set(base_data[0]["id"].map(_id))
    base_review_ids = set(base_data[2]["id"].map(_id))
    for snapshot, data in snapshot_data[1:]:
        listing_ids = set(data[0]["id"].map(_id))
        review_ids = set(data[2]["id"].map(_id))
        listing_diff = len(base_listing_ids.symmetric_difference(listing_ids))
        review_diff = len(base_review_ids.symmetric_difference(review_ids))
        rows.append({"category": "cross_snapshot", "snapshot": snapshot, "dataset": "listings", "rule": "listing_id_set_consistency", "status": "PASS" if listing_diff == 0 else "FAIL", "observed": listing_diff, "expected": 0, "details": f"so với {base_snapshot}"})
        rows.append({"category": "cross_snapshot", "snapshot": snapshot, "dataset": "reviews", "rule": "review_id_set_consistency", "status": "PASS" if review_diff == 0 else "FAIL", "observed": review_diff, "expected": 0, "details": f"so với {base_snapshot}"})
    return rows


def process_all_snapshots(config: dict[str, Any], project_root: Path, listing_assignment: pd.DataFrame, review_assignment: pd.DataFrame, bounds: pd.DataFrame) -> list[dict[str, Any]]:
    """Process all configured snapshots and collect QA results."""
    raw_dir = _resolve_path(project_root, config["raw_dir"])
    output_dir = _resolve_path(project_root, config["processed_dir"])
    report: list[dict[str, Any]] = []
    snapshot_data: list[tuple[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = []
    for snapshot_index, snapshot_value in enumerate(config["snapshots"]):
        snapshot = str(snapshot_value)
        filtered, stats = filter_snapshot(raw_dir, snapshot, listing_assignment, review_assignment)
        snapshot_data.append((snapshot, filtered))
        save_processed_data(filtered, output_dir, snapshot)
        flagged_listings = apply_listing_flags(filtered[0])
        coordinate = check_analysis_area_coordinates(flagged_listings, bounds)
        flagged_listings[COORDINATE_RULE] = coordinate["flags"].astype(bool)
        flagged = (flagged_listings, apply_calendar_flags(filtered[1]), apply_review_flags(filtered[2]))
        save_flagged_data(flagged, output_dir, snapshot)
        for dataset, frame in zip(("listings", "calendar", "reviews"), flagged):
            report.extend(_row_report(snapshot, dataset, frame))
        for dataset, values in stats.items():
            report.append({"category": "filter", "snapshot": snapshot, "dataset": dataset, "rule": "filter_summary", "status": "INFO", "observed": values, "expected": "", "details": "Số dòng raw và sau lọc"})
        report.extend(run_dataset_level_qa(snapshot, snapshot_index, filtered, listing_assignment, review_assignment))
        report.append({"category": "coordinate", "snapshot": snapshot, "dataset": "listings", "rule": COORDINATE_RULE, "status": "PASS" if coordinate["invalid"] == 0 else "FAIL", "observed": coordinate["invalid"], "expected": 0, "details": str({key: value for key, value in coordinate.items() if key != "flags"})})
    report.extend(run_cross_snapshot_qa(snapshot_data))
    return report


def save_qa_report(rows: list[dict[str, Any]], project_root: Path) -> Path:
    """Write a human-readable QA report in reports and processed."""
    report = pd.DataFrame(rows)
    report_path = project_root / "reports" / "qa_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, encoding="utf-8")
    report.to_csv(project_root / "data" / "processed" / "qa_report.csv", index=False, encoding="utf-8")
    return report_path


def run_qa(project_root: Path, config_path: Path, sheet_dir: Path | None = None) -> pd.DataFrame:
    """Run the complete assignment filtering and QA workflow."""
    config = load_config(config_path)
    raw_dir = _resolve_path(project_root, config["raw_dir"])
    paths = copy_assignment_sheets(sheet_dir, raw_dir) if sheet_dir else download_assignment_sheets(raw_dir)
    listing, review, area = load_assignment_sheets(paths)
    assignment_report = validate_assignments(listing, review, area)
    listing_assignment, review_assignment = get_group_assignments(listing, review)
    bounds = load_region_bounds(area, str(config["city"]))
    report = assignment_report + process_all_snapshots(config, project_root, listing_assignment, review_assignment, bounds)
    save_qa_report(report, project_root)
    return pd.DataFrame(report)


def main() -> int:
    """Parse CLI arguments and orchestrate the workflow."""
    parser = argparse.ArgumentParser(description="Lọc và kiểm tra chất lượng dữ liệu Airbnb.")
    parser.add_argument("--project-root", type=Path, default=find_project_root())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sheet-dir", type=Path, help="Dùng sheet local thay vì tải Google Sheets")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = (args.config or root / "configs" / "santiago.yaml").resolve()
    report = run_qa(root, config_path, args.sheet_dir.resolve() if args.sheet_dir else None)
    failures = int(report["status"].isin({"FAIL", "ERROR"}).sum()) if not report.empty else 0
    snapshots = report["snapshot"].replace("", pd.NA).dropna().nunique() if not report.empty else 0
    print(f"Đã xử lý {snapshots} snapshot; FAIL/ERROR: {failures}; báo cáo: reports/qa_report.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())