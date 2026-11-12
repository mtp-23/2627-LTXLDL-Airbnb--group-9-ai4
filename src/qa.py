from pathlib import Path
import gzip
import os
import pandas as pd
import requests
import yaml

SHEET_ID = "1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA"
DATASET_ID = "DSE2049-C04-G09-santiago"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "phan_cong"
FLAGGED_DIR = PROJECT_ROOT / "data" / "processed" / "flagged"
REPORT_DIR = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "configs" / "santiago.yaml"

SHEETS = {
    "listing": 454441734,
    "review": 2096394307,
    "area": 903142397,
}
TABLE_FILES = {
    "listings": ("listings.csv.gz", ["id"]),
    "calendar": ("calendar.csv.gz", ["listing_id", "date"]),
    "reviews": ("reviews.csv.gz", ["id"]),
}

REQUIRED_COLUMNS = {
    "listings": {
        "id",
        "neighbourhood_cleansed",
        "room_type",
        "number_of_reviews",
        "price",
        "latitude",
        "longitude",
    },
    "calendar": {
        "listing_id",
        "date",
        "available",
    },
    "reviews": {
        "id",
        "date",
        "comments",
    },
}

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
        "name": "missing_comments",
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
    {
        "name": "invalid_area",
        "condition": "tọa độ của id không khớp tọa độ vùng",
        "reason": "không thể vẽ biểu đồ vùng chính xác",
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

def replace(temp_path, output_path):
    os.replace(temp_path, output_path)

def _write_csv(data, output_path, compressed=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        data.to_csv(
            temp_path,
            index=False,
            encoding="utf-8",
            compression="gzip" if compressed else None,)
        replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def _read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,)

def _read_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file cấu hình: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("File YAML không chứa cấu hình hợp lệ")
    return config

def _validate_snapshots(config):
    snapshots = config.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 4:
        raise ValueError(
            "Cấu hình phải chứa đúng 4 snapshot")
    if len(set(snapshots)) != 4:
        raise ValueError("Danh sách snapshot bị trùng")
    if any(not isinstance(s, str) or not s.strip() for s in snapshots):
        raise ValueError("Tên snapshot không hợp lệ")
    return snapshots

def _validate_columns(data, required, path):
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} thiếu các cột: {sorted(missing)}")

def download_sheet(name, gid):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    url = (f"https://docs.google.com/spreadsheets/d/"f"{SHEET_ID}/export?format=csv&gid={gid}")
    output_path = OUTPUT_DIR / f"{name}.csv.gz"
    temp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        response = requests.get(
            url,
            stream=True,
            timeout=60,)
        response.raise_for_status()
        downloaded = 0
        with gzip.open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
        if downloaded == 0:
            raise ValueError(f"Google Sheets trả về dữ liệu rỗng: {name}")
        replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_assignments():
    print("Đang phân công nhiệm vụ!")
    for name, gid in SHEETS.items():
        path = OUTPUT_DIR / f"{name}.csv.gz"
        if not path.is_file():
            download_sheet(name, gid)

    listing = _read_csv(OUTPUT_DIR / "listing.csv.gz")
    review = _read_csv(OUTPUT_DIR / "review.csv.gz")

    _validate_columns(listing,{"Mã dữ liệu", "listing_id", "analysis_area"},"listing.csv.gz",)
    _validate_columns(review,{"Mã dữ liệu", "review_id"},"review.csv.gz",)

    listing = listing.loc[listing["Mã dữ liệu"].str.strip().eq(DATASET_ID)].copy()
    review = review.loc[review["Mã dữ liệu"].str.strip().eq(DATASET_ID)].copy()

    listing["listing_id"] = listing["listing_id"].str.strip()
    listing["analysis_area"] = listing["analysis_area"].str.strip()
    review["review_id"] = review["review_id"].str.strip()

    listing = listing.loc[listing["listing_id"].ne("")]
    review = review.loc[review["review_id"].ne("")]

    listing_ids = set(listing["listing_id"])
    review_ids = set(review["review_id"])

    if not listing_ids or not review_ids:
        raise ValueError(f"Không tìm thấy phân công hợp lệ cho {DATASET_ID}")
    return listing_ids, review_ids

def phan_cong(config):
    snapshots = _validate_snapshots(config)
    listing_ids, review_ids = _load_assignments()
    source_names = {
        "listings": ("listings.csv.gz","id",listing_ids,),
        "calendar": ("calendar.csv.gz","listing_id",listing_ids,),
        "reviews": ("reviews.csv.gz","id",review_ids,),
    }

    for snapshot in snapshots:
        source_dir = PROJECT_ROOT / config["raw_dir"] / snapshot
        output_dir = OUTPUT_DIR / snapshot
        output_dir.mkdir(parents=True, exist_ok=True)

        for table, (filename, key, assigned_ids) in source_names.items():
            source_path = source_dir / filename
            source = _read_csv(source_path)

            if key not in source.columns:
                raise ValueError(f"{source_path} thiếu cột khóa '{key}'")
            
            mask = (source[key].astype("string").str.strip().isin(assigned_ids))
            filtered = source.loc[mask].copy()
            output_path = output_dir / filename

            _write_csv(filtered,output_path,compressed=True,)

def _is_missing(series):
    values = (series.astype("string").str.strip().str.lower())
    return (series.isna() | values.isin(["", "none", "null", "nan"]))


def _numeric_price(series):
    cleaned = (series.astype("string").str.replace(r"[$,]", "", regex=True).str.strip())
    return pd.to_numeric(cleaned,errors="coerce",)


def _numeric_coordinate(series):
    cleaned = (series.astype("string").str.strip().str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned,errors="coerce",)


def _add_flag(flags, name, values):
    flags[name] = (pd.Series(values, index=flags.index).fillna(False).astype(bool))


def _new_flags(data, keys):
    flags = data[keys].copy()
    for key in keys:
        flags[key] = flags[key].astype(str)
    return flags

def _read_inputs(config):
    inputs = {}
    for snapshot in _validate_snapshots(config):
        inputs[snapshot] = {}
        for table, (filename, keys) in TABLE_FILES.items():
            path = OUTPUT_DIR / snapshot / filename
            data = _read_csv(path)
            required = (REQUIRED_COLUMNS[table] | set(keys))
            _validate_columns(data,required,path,)
            inputs[snapshot][table] = data
    return inputs

def _read_assignment_areas():
    assignment = _read_csv(OUTPUT_DIR / "listing.csv.gz")
    required = {"Mã dữ liệu","listing_id","analysis_area",}
    _validate_columns(assignment,required,"listing.csv.gz",)

    assignment = assignment.loc[assignment["Mã dữ liệu"].str.strip().eq(DATASET_ID),["listing_id", "analysis_area"],].copy()
    assignment["listing_id"] = (assignment["listing_id"].str.strip())
    assignment["analysis_area"] = (assignment["analysis_area"].str.strip())
    assignment = assignment.loc[assignment["listing_id"].ne("")]
    area_count = (assignment.groupby("listing_id")["analysis_area"].nunique())

    if area_count.gt(1).any():
        raise ValueError("Một listing_id được phân công vào nhiều analysis_area")
    return (assignment.drop_duplicates("listing_id").set_index("listing_id")["analysis_area"])


def _read_area_bounds(config):
    path = OUTPUT_DIR / "area.csv.gz"
    area = _read_csv(path)
    required = {"Thành phố","analysis_area","lon_min","lat_min","lon_max","lat_max",}
    _validate_columns(area, required, path)

    area = area.loc[area["Thành phố"].str.strip().eq(str(config["city"]).strip())].copy()
    if area.empty:
        raise ValueError(f"Không tìm thấy khu vực của thành phố {config['city']}")

    area["analysis_area"] = (area["analysis_area"].str.strip())
    if area["analysis_area"].eq("").any():
        raise ValueError("Sheet area có analysis_area bị thiếu")
    if area["analysis_area"].duplicated().any():
        raise ValueError("Sheet area có analysis_area bị trùng")

    columns = ["lon_min","lat_min","lon_max","lat_max",]
    for column in columns:
        area[column] = _numeric_coordinate(area[column])
    if area[columns].isna().any().any():
        raise ValueError("Sheet area có tọa độ min-max không hợp lệ")
    if (area["lon_min"].gt(area["lon_max"]).any() or area["lat_min"].gt(area["lat_max"]).any()):
        raise ValueError("Tọa độ min-max của area không hợp lệ")

    return area.set_index("analysis_area")

def _invalid_area_coordinates(listings,assignment_areas,area_bounds,):
    listing_ids = (listings["id"].astype(str).str.strip())
    assigned_areas = listing_ids.map(assignment_areas)
    latitude = _numeric_coordinate(listings["latitude"])
    longitude = _numeric_coordinate(listings["longitude"])

    lat_min = assigned_areas.map(area_bounds["lat_min"])
    lat_max = assigned_areas.map(area_bounds["lat_max"])
    lon_min = assigned_areas.map(area_bounds["lon_min"])
    lon_max = assigned_areas.map(area_bounds["lon_max"])

    return (
        assigned_areas.isna()
        | latitude.isna()
        | longitude.isna()
        | lat_min.isna()
        | lat_max.isna()
        | lon_min.isna()
        | lon_max.isna()
        | latitude.lt(lat_min)
        | latitude.gt(lat_max)
        | longitude.lt(lon_min)
        | longitude.gt(lon_max)
    )

def _build_flag_frames():
    config = _read_config()
    inputs = _read_inputs(config)
    assignment_areas = _read_assignment_areas()
    area_bounds = _read_area_bounds(config)

    frames = {
        snapshot: {table: _new_flags(data,TABLE_FILES[table][1],) for table, data in tables.items()} 
        for snapshot, tables in inputs.items()
    }

    review_snapshot_count = {}
    for tables in inputs.values():
        review_ids = tables["reviews"]["id"]
        for review_id in review_ids.drop_duplicates():
            if not review_id.strip():
                continue
            review_snapshot_count[review_id] = (review_snapshot_count.get(review_id, 0) + 1)

    for snapshot, tables in inputs.items():
        print(f"Đang gắn cờ mốc {snapshot}!")
        listings = tables["listings"]
        calendar = tables["calendar"]
        reviews = tables["reviews"]
        listing_flags = frames[snapshot]["listings"]
        calendar_flags = frames[snapshot]["calendar"]
        review_flags = frames[snapshot]["reviews"]

        # missing
        missing_columns = {
            "missing_neigh_cleansed": "neighbourhood_cleansed",
            "missing_room_type": "room_type",
            "missing_number_of_reviews": "number_of_reviews",
            "missing_price": "price",
        }

        for rule_name, column in missing_columns.items():
            _add_flag(listing_flags,rule_name,_is_missing(listings[column]),)

        _add_flag(review_flags,"missing_review_date",_is_missing(reviews["date"]),)
        _add_flag(review_flags,"missing_comments",_is_missing(reviews["comments"]),)

        # invalid
        review_count = (listings["number_of_reviews"].astype("string").str.strip())
        missing_review_count = _is_missing(listings["number_of_reviews"])
        _add_flag(listing_flags,"invalid_reviews_num",(~missing_review_count & ~review_count.str.fullmatch(r"\d+",na=False,)),)

        accepted_room_types = {
            "Entire home/apt",
            "Private room",
            "Shared room",
            "Hotel room",
        }
        room_type = (listings["room_type"].astype("string").str.strip())
        _add_flag(listing_flags,"invalid_room_type",(~_is_missing(listings["room_type"]) & ~room_type.isin(accepted_room_types)),)

        price = _numeric_price(listings["price"])
        _add_flag(listing_flags,"invalid_price",(~_is_missing(listings["price"]) & price.isna()),)

        calendar_date_missing = _is_missing(calendar["date"])
        calendar_date = pd.to_datetime(calendar["date"],errors="coerce",)
        _add_flag(calendar_flags,"invalid_calendar_date",(~calendar_date_missing & calendar_date.isna()),)

        available = (calendar["available"].astype("string").str.strip().str.lower())
        _add_flag(calendar_flags,"invalid_calendar_available",(~_is_missing(calendar["available"]) & ~available.isin({"t", "f"})),)

        review_date_missing = _is_missing(reviews["date"])
        review_date = pd.to_datetime(reviews["date"],errors="coerce",)
        _add_flag(review_flags,"invalid_review_date",(~review_date_missing & review_date.isna()),)
        _add_flag(listing_flags,"invalid_area",_invalid_area_coordinates(listings,assignment_areas,area_bounds,),)

        # outlier
        valid_price = price.dropna()
        if valid_price.empty:
            outlier_price = pd.Series(False,index=listings.index,)
        else:
            q1, q3 = valid_price.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_price = ((price < lower) | (price > upper))
        _add_flag(listing_flags,"outlier_price",outlier_price,)

        # duplicates
        _add_flag(listing_flags,"dup_listing_id",listings["id"].duplicated(keep=False),)
        _add_flag(calendar_flags,"dup_cal_key",calendar.duplicated(subset=["listing_id", "date"],keep=False,),)
        _add_flag(review_flags,"dup_review_id_in_snapshot",reviews["id"].duplicated(keep=False),)
        _add_flag(review_flags,"dup_review_id_across_snapshots",(reviews["id"].map(review_snapshot_count).fillna(0).gt(1)),)

        # Calendar completeness
        row_counts = (calendar.groupby("listing_id").size())
        unique_days = (calendar.groupby("listing_id")["date"].nunique())
        row_mismatch = (calendar["listing_id"].map(row_counts).ne(365))
        day_mismatch = (calendar["listing_id"].map(unique_days).ne(365))
        _add_flag(calendar_flags,"calendar_days_per_listing_mismatch",(row_mismatch | day_mismatch),)

    return frames

def _write_flag_frames(frames):
    for snapshot, tables in frames.items():
        output_dir = FLAGGED_DIR / snapshot
        for table, flags in tables.items():
            output_path = (output_dir / f"{table}_flags.csv.gz")
            _write_csv(flags,output_path,compressed=True,)

def run_qa():
    frames = _build_flag_frames()
    _write_flag_frames(frames)

def main():
    try:
        config = _read_config()
        phan_cong(config)
        run_qa()
        print("Đã xong!")
        return 0
    except Exception as error:
        print(f"Lỗi: {type(error).__name__}: {error}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())