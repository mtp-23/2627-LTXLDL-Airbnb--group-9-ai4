
import pandas as pd
from pathlib import Path

SPREADSHEET_ID = "1O0sSfiparDq_CwPpXjrO4w3TOpESWLMD37IT1sZJOUY"

GID_REVIEW = "1706381649"
GID_LISTING = "2108303386"

OUTPUT_DIR = Path("data/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_sheet(gid, output_name):
    """Tải một tab Google Sheet và lưu thành CSV."""
    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SPREADSHEET_ID}/export?format=csv&gid={gid}"
    )

    print(f"Đang tải tab vào {output_name}...")

    df = pd.read_csv(url)

    output_path = OUTPUT_DIR / output_name

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Đã lưu: {output_path}")
    print(f"Số dòng: {len(df)}")
    print(f"Các cột: {df.columns.tolist()}")
    print()

    return df


def main():
    review_df = download_sheet(
        GID_REVIEW,
        "review.csv"
    )

    listing_df = download_sheet(
        GID_LISTING,
        "listing.csv"
    )

    print("Hoàn tất tải dữ liệu.")
    print("Review:", review_df.shape)
    print("Listing:", listing_df.shape)


if __name__ == "__main__":
    main()
