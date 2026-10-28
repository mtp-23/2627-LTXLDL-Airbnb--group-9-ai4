import argparse
import csv
import gzip
import json
import sys
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit

import requests
import yaml

MAX_RETRIES = 3
TIMEOUT = 60  # giây
CHUNK = 1024 * 1024  # 1 MB
HEADERS = {"User-Agent": "Mozilla/5.0 (student-project; Inside Airbnb downloader)"}
SNAPSHOT_COUNT = 4
FILE_KEYS = {
    "listings": ("id",),
    "calendar": ("listing_id", "date"),
    "reviews": ("id",),
}


# Đọc cấu hình 
def find_config(arg: str | None) -> Path:
    if arg:
        if not Path(arg).exists():
            sys.exit(f"Không tìm thấy file cấu hình: {Path(arg)}")
        return Path(arg)

    cf = Path(__file__).resolve().parent.parent
    found = list((cf / "configs").glob("*.yaml"))

    if not found:
        sys.exit("Không tìm thấy .yaml")
    return found[0]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "snapshots", "download", "raw_dir"):
        if key not in cfg:
            sys.exit(f"Thiếu khóa '{key}' trong configs")
    for key in ("URL_CSV", "URL_GEOJSON", "f_type"):
        if key not in cfg["download"]:
            sys.exit(f"Thiếu khóa '{key}' trong configs")

    snapshots = cfg["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != SNAPSHOT_COUNT:
        sys.exit(f"Cần đúng {SNAPSHOT_COUNT} mốc trong 'snapshots' của configs")
    normalized_snapshots = [str(date) for date in snapshots]
    if len(set(normalized_snapshots)) != SNAPSHOT_COUNT:
        sys.exit(f"Các mốc trong 'snapshots' phải khác nhau trong configs")

    file_types = cfg["download"]["f_type"]
    if not isinstance(file_types, list) or set(file_types) != set(FILE_KEYS):
        sys.exit("[download][f_type] phải bao gồm listings, calendar và reviews.")

    url_template = cfg["download"]["URL_CSV"]
    parsed_url = urlsplit(url_template)
    if parsed_url.scheme != "https" or parsed_url.hostname != "data.insideairbnb.com":
        sys.exit("URL_CSV phải dùng HTTPS và host data.insideairbnb.com.")
    if parsed_url.path != "/{data_root}/{date}/data/{f_type}.csv.gz":
        sys.exit("URL_CSV phải theo mẫu .../{data_root}/{date}/data/{f_type}.csv.gz.")
    geojson_url = urlsplit(cfg["download"]["URL_GEOJSON"])
    if geojson_url.scheme != "https" or geojson_url.hostname != "data.insideairbnb.com":
        sys.exit("URL_GEOJSON phải dùng HTTPS và host data.insideairbnb.com.")
    if geojson_url.path != "/{data_root}/{date}/visualisations/neighbourhoods.geojson":
        sys.exit("URL_GEOJSON phải theo mẫu .../{data_root}/{date}/visualisations/neighbourhoods.geojson.")
    return cfg


# Tạo danh sách tải
def build_jobs(cfg: dict, out_root: Path):
    dl = cfg["download"]
    data_root = cfg["data_root"].strip("/")
    jobs = []

    for date in cfg["snapshots"]:
        date = str(date)  # phòng trường hợp YAML tự hiểu thành kiểu ngày
        folder = out_root / date

        for f_type in FILE_KEYS:
            url = dl["URL_CSV"].format(data_root=data_root, date=date, f_type=f_type)
            jobs.append((url, folder / f"{f_type}.csv.gz"))

        geojson_url = dl["URL_GEOJSON"].format(data_root=data_root, date=date)
        jobs.append((geojson_url, folder / "neighbourhoods.geojson"))

    return jobs


def validate_csv(path: Path, file_type: str) -> str | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), None)
    except (OSError, EOFError, UnicodeError, csv.Error, zlib.error) as e:
        return f"không đọc được CSV gzip ({e})"

    if not header:
        return "tệp rỗng hoặc không có header"

    columns = {column.strip() for column in header}
    missing = [key for key in FILE_KEYS[file_type] if key not in columns]
    if missing:
        return f"thiếu cột khóa: {', '.join(missing)}"
    return None


def validate_geojson(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        return f"không đọc được GeoJSON ({e})"

    if not isinstance(document, dict):
        return "Nội dung GeoJSON phải là JSON"
    if document.get("type") != "FeatureCollection":
        return "GeoJSON không có type FeatureCollection"
    if not isinstance(document.get("features"), list):
        return "GeoJSON không có danh sách features"
    return None


# Tải file
def download_file(url: str, dest: Path, file_type: str | None = None, force: bool = False) -> str:
    if dest.exists() and not force:
        if dest.stat().st_size > 0:
            invalid_reason = (validate_csv(dest, file_type)
                              if file_type else validate_geojson(dest))
            if invalid_reason is None:
                print(f"\nĐã có tệp: {dest}")
                return "skipped"
            print(f"\n{dest.name}: {invalid_reason}; sẽ tải lại")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    print(f"\n[404] không tồn tại: {url}")
                    return "missing"
                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {dest.name}: {done / 1e6:6.1f}/{total / 1e6:.1f} MB",
                                  end="", flush=True)
                print()

            if tmp.stat().st_size == 0:
                raise OSError("máy chủ trả về tệp rỗng")
            invalid_reason = (validate_csv(tmp, file_type)
                              if file_type else validate_geojson(tmp))
            if invalid_reason is not None:
                raise OSError(f"tệp tải về không hợp lệ: {invalid_reason}")

            tmp.replace(dest)  # chỉ đổi tên khi tải xong hoàn toàn
            print(f"\n[done] {dest}")
            return "done"

        except (requests.RequestException, OSError) as e:
            print(f"\n[lỗi] lần {attempt}/{MAX_RETRIES}: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    return "failed"


# Main
def main() -> int:
    ap = argparse.ArgumentParser(description="Tải dữ liệu Inside Airbnb theo file config YAML")
    ap.add_argument("--config", help="Đường dẫn yaml có dạng configs/*.yaml")
    ap.add_argument("--force", action="store_true", help="Ghi đè tệp đã có")
    ap.add_argument("--test", action="store_true", help="Chỉ in URL và nơi lưu, không tải")
    args = ap.parse_args()

    cfg_path = find_config(args.config).resolve()
    cfg = load_config(cfg_path)
    out_root = Path(cfg["raw_dir"])

    jobs = build_jobs(cfg, out_root)
    stats = {"done": 0, "skipped": 0, "missing": 0, "failed": 0}

    last_folder = None
    for url, dest in jobs:
        if dest.parent != last_folder:
            last_folder = dest.parent
            print("===============")
            print(f"Mốc: {dest.parent.name}")
        if args.test:
            print(f"\n{dest.name}\n   -> {dest}")
            continue
        file_type = (dest.name.removesuffix(".csv.gz")
                     if dest.name.endswith(".csv.gz") else None)
        stats[download_file(url, dest, file_type, args.force)] += 1

    if args.test:
        print(f"\nTổng cộng {len(jobs)} tệp.")
        return 0

    print("\nHoàn thành")
    print(f"\n  Tải mới: {stats['done']}\n  Đã có: {stats['skipped']}"
          f"\n  Không tìm thấy: {stats['missing']}\n    Lỗi: {stats['failed']}")
    if stats["missing"]:
        print("\nKiểm tra lại mốc")
    return 1 if stats["failed"] or stats["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())