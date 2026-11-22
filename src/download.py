"""Tải dữ liệu thô và kiểm tra toàn vẹn tệp tải về.

Nhiệm vụ duy nhất: lấy tệp từ Inside Airbnb / Google Sheets về đĩa và đảm bảo
tệp đọc được, có cột khóa. Không đọc cấu hình, không lọc, không làm sạch.
"""
from __future__ import annotations

import csv
import gzip
import json
import time
import zlib
from pathlib import Path
from urllib.parse import urlsplit

import requests

MAX_RETRIES = 3
TIMEOUT = 60  # giây
CHUNK = 1024 * 1024
HEADERS = {"User-Agent": "Mozilla/5.0 (student-project; Inside Airbnb downloader)"}
HOST = "data.insideairbnb.com"
FILE_KEYS = {
    "listings": ("id",),
    "calendar": ("listing_id", "date"),
    "reviews": ("id",),
}
CSV_PATH = "/{data_root}/{date}/data/{f_type}.csv.gz"
GEOJSON_PATH = "/{data_root}/{date}/visualisations/neighbourhoods.geojson"


def _check_url(template: str, expected_path: str, label: str) -> None:
    parsed = urlsplit(template)
    if parsed.scheme != "https" or parsed.hostname != HOST:
        raise ValueError(f"{label} phải dùng HTTPS và host {HOST}.")
    if parsed.path != expected_path:
        raise ValueError(f"{label} phải theo mẫu .../{expected_path.lstrip('/')}.")


def build_jobs(download_cfg: dict, data_root: str, snapshots: list[str], out_root: Path):
    """Danh sách (url, đường dẫn đích, loại tệp); loại tệp None nghĩa là GeoJSON."""
    file_types = download_cfg.get("f_type")
    if not isinstance(file_types, list) or set(file_types) != set(FILE_KEYS):
        raise ValueError("download.f_type phải gồm listings, calendar và reviews.")
    csv_url = download_cfg.get("url_csv")
    if not csv_url:
        raise ValueError("Thiếu download.url_csv trong cấu hình.")
    _check_url(csv_url, CSV_PATH, "url_csv")
    geojson_url = download_cfg.get("url_geojson")  # tùy chọn: chỉ cần cho bản đồ
    if geojson_url:
        _check_url(geojson_url, GEOJSON_PATH, "url_geojson")

    data_root = data_root.strip("/")
    jobs = []
    for date in snapshots:
        folder = out_root / str(date)
        for f_type in FILE_KEYS:
            url = csv_url.format(data_root=data_root, date=date, f_type=f_type)
            jobs.append((url, folder / f"{f_type}.csv.gz", f_type))
        if geojson_url:
            url = geojson_url.format(data_root=data_root, date=date)
            jobs.append((url, folder / "neighbourhoods.geojson", None))
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
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        return "GeoJSON không có type FeatureCollection"
    if not isinstance(document.get("features"), list):
        return "GeoJSON không có danh sách features"
    return None


def download_file(url: str, dest: Path, file_type: str | None, force: bool = False) -> str:
    """Trả về 'skipped' | 'done' | 'missing' (404) | 'failed'."""
    validate = (lambda p: validate_csv(p, file_type)) if file_type else validate_geojson
    if dest.exists() and not force and dest.stat().st_size > 0 and validate(dest) is None:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size == 0:
                raise OSError("máy chủ trả về tệp rỗng")
            reason = validate(tmp)
            if reason is not None:
                raise OSError(f"tệp tải về không hợp lệ: {reason}")
            tmp.replace(dest)  # chỉ đổi tên khi tải xong hoàn toàn
            return "done"
        except (requests.RequestException, OSError) as e:
            print(f"[lỗi] {dest.name} lần {attempt}/{MAX_RETRIES}: {e}")
            tmp.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    return "failed"


def download_all(download_cfg: dict, data_root: str, snapshots: list[str],
                 out_root: Path, force: bool = False) -> dict:
    """Tải đủ các mốc; trả về thống kê {'done','skipped','missing','failed'}."""
    stats = {"done": 0, "skipped": 0, "missing": 0, "failed": 0}
    current = None
    for url, dest, file_type in build_jobs(download_cfg, data_root, snapshots, out_root):
        if dest.parent.name != current:
            current = dest.parent.name
            print(f"Đang tải mốc {current}!")
        stats[download_file(url, dest, file_type, force)] += 1
    return stats


def fetch_sheet(sheet_id: str, gid: int | str) -> bytes:
    """Tải một sheet Google dưới dạng CSV (bytes); người gọi quyết định lưu ở đâu."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    if not r.content:
        raise OSError(f"Google Sheets trả về dữ liệu rỗng (gid={gid})")
    return r.content