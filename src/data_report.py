
from pathlib import Path
from datetime import date
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path(
    "reports/check_DSE2049-C04-G09-santiago.csv"
)

MA_DU_LIEU = "DSE2049-C04-G09-santiago"

SNAPSHOTS = [
    (1, "2025-09-27"),
    (2, "2026-03-30"),
    (3, "2026-06-29"),
    (4, "2026-09-23"),
]


def find_file(folder, filename):
    """Tìm file CSV hoặc CSV.GZ trong một thư mục."""
    for suffix in [".csv.gz", ".csv"]:
        path = folder / f"{filename}{suffix}"
        if path.exists():
            return path

    return None


def read_data(path):
    """Đọc CSV/CSV.GZ; ô trống được pandas tính là NaN."""
    return pd.read_csv(path, low_memory=False)


def file_size(path):
    """Trả về dung lượng file theo byte."""
    if path is None:
        return None

    return path.stat().st_size


def get_source_sizes(snapshot):
    """Lấy dung lượng ba file nguồn trong data/raw."""
    folder = RAW_DIR / snapshot

    files = {
        "listings": find_file(folder, "listings"),
        "calendar": find_file(folder, "calendar"),
        "reviews": find_file(folder, "reviews"),
    }

    sizes = {
        "listings_bytes": file_size(files["listings"]),
        "calendar_bytes": file_size(files["calendar"]),
        "reviews_bytes": file_size(files["reviews"]),
    }

    # Có đủ cả ba file nguồn hay không
    all_downloaded = all(path is not None for path in files.values())

    # Chỉ xác nhận có file; không khẳng định file đọc được
    download_status = "co" if all_downloaded else "khong co"

    return sizes, download_status


def find_column(df, column_name):
    target = column_name.strip().casefold()

    for col in df.columns:
        if str(col).strip().casefold() == target:
            return col

    raise ValueError(
        f"Không tìm thấy cột '{column_name}'. "
        f"Các cột hiện có: {list(df.columns)}"
    )


def check_snapshot(moc, snapshot):
    folder = PROCESSED_DIR / snapshot

    # Đọc dữ liệu đã lọc
    paths = {
        "listings": find_file(folder, "listings"),
        "calendar": find_file(folder, "calendar"),
        "reviews": find_file(folder, "reviews"),
    }

    missing = [
        name for name, path in paths.items()
        if path is None
    ]

    if missing:
        raise FileNotFoundError(
            f"{snapshot} thiếu file processed: {missing}"
        )

    listings = read_data(paths["listings"])
    calendar = read_data(paths["calendar"])
    reviews = read_data(paths["reviews"])

    # Tên cột cần dùng
    room_type_col = find_column(listings, "room_type")
    no_reviews_col = find_column(listings, "number_of_reviews")
    price_col = find_column(listings, "price")

    # Đếm loại phòng
    room_type = (
        listings[room_type_col]
        .astype("string")
        .str.strip()
    )

    entire_home = int(
        (room_type == "Entire home/apt").sum()
    )

    private_room = int(
        (room_type == "Private room").sum()
    )

    # Đếm listing không có review
    number_of_reviews = pd.to_numeric(
        listings[no_reviews_col],
        errors="coerce",
    )

    no_reviews = int(
        (number_of_reviews == 0).sum()
    )

    # Đếm giá bị thiếu
    missing_price = int(
        listings[price_col].isna().sum()
    )

    # Số cột có ít nhất 90% giá trị thiếu
    missing_ratio = listings.isna().mean()
    columns_nan_90 = int(
        (missing_ratio >= 0.90).sum()
    )

    # Dung lượng file nguồn và tình trạng có file
    sizes, download_status = get_source_sizes(snapshot)

    result = {
        "ma_du_lieu": MA_DU_LIEU,
        "moc": moc,
        "ngay": snapshot,
        "tai_ve_ok": download_status,
        "listings_bytes": sizes["listings_bytes"],
        "calendar_bytes": sizes["calendar_bytes"],
        "reviews_bytes": sizes["reviews_bytes"],
        "listings_dong": len(listings),
        "calendar_dong": len(calendar),
        "reviews_dong": len(reviews),
        "entire_home": entire_home,
        "private_room": private_room,
        "khong_co_review": no_reviews,
        "thieu_price": missing_price,
        "so_cot_listings": len(listings.columns),
        "so_cot_nan_90": columns_nan_90,
    }

    return result


def main():
    results = []

    for moc, snapshot in SNAPSHOTS:
        results.append(check_snapshot(moc, snapshot))

    report = pd.DataFrame(results)

    columns = [
        "ma_du_lieu",
        "moc",
        "ngay",
        "tai_ve_ok",
        "listings_bytes",
        "calendar_bytes",
        "reviews_bytes",
        "listings_dong",
        "calendar_dong",
        "reviews_dong",
        "entire_home",
        "private_room",
        "khong_co_review",
        "thieu_price",
        "so_cot_listings",
        "so_cot_nan_90",
    ]

    report = report[columns]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã cập nhật báo cáo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()