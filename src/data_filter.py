from pathlib import Path
import argparse
import pandas as pd


# Các thư mục mặc định, tính từ thư mục gốc của project
RAW_DIR = Path("data/raw")
SAMPLES_DIR = Path("data/samples")
OUTPUT_DIR = Path("data/processed")

LISTING = SAMPLES_DIR / "listing.csv"
REVIEW = SAMPLES_DIR / "review.csv"

EXPECTED_LISTINGS = 1000
EXPECTED_REVIEWS = 5000


def read_csv(path):
    return pd.read_csv(
        path,
        compression="infer",
        dtype={"id": "string", "listing_id": "string",
               "review_id": "string"},
        low_memory=False,
    )


def normalize_id(series):
    """Chuẩn hóa ID để tránh lệch kiểu int/float/string."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def find_column(df, name):
    """Tìm tên cột không phân biệt chữ hoa/thường."""
    target = name.strip().casefold()

    for col in df.columns:
        if str(col).strip().lstrip("\ufeff").casefold() == target:
            return col

    raise ValueError(
        f"Không tìm thấy cột '{name}'. "
        f"Các cột hiện có: {list(df.columns)}"
    )


def clean_headers(df):
    df.columns = [
        str(col).strip().lstrip("\ufeff")
        for col in df.columns
    ]
    return df


def load_assignments():
    if not LISTING.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {LISTING}"
        )

    if not REVIEW.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {REVIEW}"
        )

    listing_map = clean_headers(read_csv(LISTING))
    review_map = clean_headers(read_csv(REVIEW))

    listing_id_col = find_column(listing_map, "listing_id")
    area_col = find_column(listing_map, "analysis_area")

    listing_map = listing_map[
        [listing_id_col, area_col]
    ].copy()

    listing_map.columns = ["listing_id", "analysis_area"]
    listing_map["listing_id"] = normalize_id(
        listing_map["listing_id"]
    )

    if listing_map["listing_id"].isna().any():
        raise ValueError("Sheet Listing có listing_id bị thiếu.")

    if listing_map["listing_id"].duplicated().any():
        raise ValueError("Sheet Listing có listing_id bị trùng.")

    if listing_map["analysis_area"].isna().any():
        raise ValueError("Sheet Listing có analysis_area bị thiếu.")

    if len(listing_map) != EXPECTED_LISTINGS:
        raise ValueError(
            f"Sheet Listing có {len(listing_map)} dòng, "
            f"yêu cầu {EXPECTED_LISTINGS}."
        )

    review_id_col = find_column(review_map, "review_id")
    review_listing_col = find_column(review_map, "listing_id")
    llm_col = find_column(review_map, "Mẫu LLM")
    manual_col = find_column(review_map, "Nhãn tay")

    review_map = review_map[
        [review_id_col, review_listing_col, llm_col, manual_col]
    ].copy()

    review_map.columns = [
        "review_id", "listing_id", "Mẫu LLM", "Nhãn tay"
    ]

    review_map["review_id"] = normalize_id(
        review_map["review_id"]
    )
    review_map["listing_id"] = normalize_id(
        review_map["listing_id"]
    )

    if review_map["review_id"].isna().any():
        raise ValueError("Sheet Review có review_id bị thiếu.")

    if review_map["review_id"].duplicated().any():
        raise ValueError("Sheet Review có review_id bị trùng.")

    if len(review_map) != EXPECTED_REVIEWS:
        raise ValueError(
            f"Sheet Review có {len(review_map)} dòng, "
            f"yêu cầu {EXPECTED_REVIEWS}."
        )

    return listing_map, review_map


def filter_listings(snapshot_dir, listing_map):
    path = snapshot_dir / "listings.csv.gz"
    if not path.exists():
        path = snapshot_dir / "listings.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file listings trong {snapshot_dir}"
        )

    df = clean_headers(read_csv(path))
    id_col = find_column(df, "id")
    df[id_col] = normalize_id(df[id_col])

    wanted_ids = set(listing_map["listing_id"])
    filtered = df[df[id_col].isin(wanted_ids)].copy()

    # Gắn analysis_area theo ID được giao
    filtered = filtered.merge(
        listing_map,
        left_on=id_col,
        right_on="listing_id",
        how="left",
        validate="many_to_one",
    )

    print(
        f"  Listings: {len(filtered)} dòng; "
        f"{filtered[id_col].nunique()} ID phân biệt; "
        f"{filtered['analysis_area'].isna().sum()} dòng thiếu area"
    )

    found_ids = set(filtered[id_col].dropna())
    missing_ids = wanted_ids - found_ids

    if missing_ids:
        print(f"  CẢNH BÁO: thiếu {len(missing_ids)} listing ID.")

    if filtered[id_col].duplicated().any():
        print("  CẢNH BÁO: dữ liệu nguồn có listing ID trùng.")

    return filtered


def filter_calendar(snapshot_dir, listing_map):
    path = snapshot_dir / "calendar.csv.gz"
    if not path.exists():
        path = snapshot_dir / "calendar.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file calendar trong {snapshot_dir}"
        )

    df = clean_headers(read_csv(path))
    id_col = find_column(df, "listing_id")
    df[id_col] = normalize_id(df[id_col])

    wanted_ids = set(listing_map["listing_id"])
    filtered = df[df[id_col].isin(wanted_ids)].copy()

    unique_ids = filtered[id_col].nunique()

    print(
        f"  Calendar: {len(filtered)} dòng; "
        f"{unique_ids} listing ID phân biệt"
    )

    if len(filtered) != EXPECTED_LISTINGS * 365:
        print(
            "  CẢNH BÁO: số dòng calendar khác "
            f"{EXPECTED_LISTINGS * 365:,}."
        )

    if filtered.duplicated([id_col, find_column(filtered, "date")]).any():
        print("  CẢNH BÁO: có listing-date bị trùng.")

    return filtered


def filter_reviews(snapshot_dir, review_map):
    path = snapshot_dir / "reviews.csv.gz"
    if not path.exists():
        path = snapshot_dir / "reviews.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file reviews trong {snapshot_dir}"
        )

    df = clean_headers(read_csv(path))
    id_col = find_column(df, "id")
    df[id_col] = normalize_id(df[id_col])

    wanted_ids = set(review_map["review_id"])
    filtered = df[df[id_col].isin(wanted_ids)].copy()

    # Gắn các cờ phân công từ sheet vào review tương ứng
    filtered = filtered.merge(
        review_map,
        left_on=id_col,
        right_on="review_id",
        how="left",
        validate="many_to_one",
    )

    print(
        f"  Reviews: {len(filtered)} dòng; "
        f"{filtered[id_col].nunique()} review ID phân biệt"
    )

    missing_ids = wanted_ids - set(filtered[id_col].dropna())
    if missing_ids:
        print(f"  CẢNH BÁO: thiếu {len(missing_ids)} review ID.")

    if filtered[id_col].duplicated().any():
        print("  CẢNH BÁO: dữ liệu nguồn có review ID trùng.")

    return filtered


def filter_snapshot(snapshot_dir, listing_map, review_map):
    """Lọc cả ba loại dữ liệu của một snapshot."""
    print(f"\nSnapshot: {snapshot_dir.name}")

    output_dir = OUTPUT_DIR / snapshot_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    listings = filter_listings(snapshot_dir, listing_map)
    calendar = filter_calendar(snapshot_dir, listing_map)
    reviews = filter_reviews(snapshot_dir, review_map)

    listings.to_csv(
        output_dir / "listings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calendar.to_csv(
        output_dir / "calendar.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reviews.to_csv(
        output_dir / "reviews.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"  Đã lưu kết quả tại: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshots",
        nargs="*",
        help="Tên snapshot cần lọc; bỏ trống để tự tìm thư mục có đủ 3 file.",
    )
    args = parser.parse_args()

    listing_map, review_map = load_assignments()

    if args.snapshots:
        snapshot_dirs = [RAW_DIR / name for name in args.snapshots]
    else:
        snapshot_dirs = sorted(
            path for path in RAW_DIR.iterdir()
            if path.is_dir()
            and any(path.glob("listings.csv*"))
            and any(path.glob("calendar.csv*"))
            and any(path.glob("reviews.csv*"))
        )

    if not snapshot_dirs:
        raise FileNotFoundError(
            f"Không tìm thấy snapshot phù hợp trong {RAW_DIR}"
        )

    for snapshot_dir in snapshot_dirs:
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy snapshot: {snapshot_dir}"
            )

        filter_snapshot(snapshot_dir, listing_map, review_map)


if __name__ == "__main__":
    main()
