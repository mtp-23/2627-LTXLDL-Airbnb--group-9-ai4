#!/usr/bin/env python3
"""
Tải dữ liệu Inside Airbnb (CC BY 4.0) theo file cấu hình YAML trong configs/.

Đọc từ config:
    data_root, snapshots, download.URL_CSV, download.URL_GEOJSON,
    download.f_type, raw_dir

Với mỗi mốc (snapshot) tải:
    - listings.csv.gz
    - calendar.csv.gz
    - reviews.csv.gz
    - neighbourhoods.geojson

Lưu tại:  <raw_dir>/<tên_thành_phố>/<mốc>/<tệp>

Cách chạy (từ thư mục gốc dự án):
    python src/download.py --config configs/santiago.yaml
    python src/download.py --force       # tải lại, ghi đè
    python src/download.py --dry-run     # chỉ in URL, không tải
"""
import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

MAX_RETRIES = 3
TIMEOUT = 60  # giây
CHUNK = 1024 * 1024  # 1 MB
HEADERS = {"User-Agent": "Mozilla/5.0 (student-project; Inside Airbnb downloader)"}


# ----------------------------- Đọc cấu hình -----------------------------
def find_config(arg: str | None) -> Path:
    """Dùng --config nếu có; nếu không thì tìm file .yaml/.yml duy nhất trong configs/."""
    if arg:
        p = Path(arg)
        if not p.exists():
            sys.exit(f"Không tìm thấy file cấu hình: {p}")
        return p

    here = Path(__file__).resolve().parent
    for root in (Path.cwd(), here, here.parent):
        found = sorted(list((root / "configs").glob("*.yaml")) +
                       list((root / "configs").glob("*.yml")))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            names = ", ".join(f.name for f in found)
            sys.exit(f"Có nhiều file cấu hình ({names}). Hãy chỉ rõ bằng --config <đường dẫn>.")
    sys.exit("Không tìm thấy file yaml trong configs/. Hãy dùng --config <đường dẫn>.")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "snapshots", "download", "raw_dir"):
        if key not in cfg:
            sys.exit(f"Thiếu khóa '{key}' trong {path}")
    for key in ("URL_CSV", "URL_GEOJSON", "f_type"):
        if key not in cfg["download"]:
            sys.exit(f"Thiếu khóa 'download.{key}' trong {path}")
    return cfg


# ----------------------------- Tạo danh sách tải -----------------------------
def build_jobs(cfg: dict, out_root: Path):
    """Trả về list (url, đường_dẫn_lưu)."""
    dl = cfg["download"]
    data_root = cfg["data_root"].strip("/")
    city_slug = data_root.split("/")[-1]  # vd: santiago
    jobs = []

    for date in cfg["snapshots"]:
        date = str(date)  # phòng trường hợp YAML tự hiểu thành kiểu ngày
        folder = out_root / city_slug / date

        for f_type in dl["f_type"]:
            url = dl["URL_CSV"].format(data_root=data_root, date=date, f_type=f_type)
            jobs.append((url, folder / f"{f_type}.csv.gz"))

        url = dl["URL_GEOJSON"].format(data_root=data_root, date=date)
        jobs.append((url, folder / "neighbourhoods.geojson"))

    return jobs


# ----------------------------- Tải 1 tệp -----------------------------
def download_file(url: str, dest: Path, force: bool = False) -> str:
    """Trả về 'ok' | 'skipped' | 'missing' | 'failed'."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [bỏ qua] đã có {dest}")
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    print(f"  [404] không tồn tại: {url}")
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

            tmp.replace(dest)  # chỉ đổi tên khi tải xong hoàn toàn
            print(f"  [ok] {dest}")
            return "ok"

        except requests.RequestException as e:
            print(f"\n  [lỗi] lần {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(2 * attempt)

    if tmp.exists():
        tmp.unlink()
    return "failed"


# ----------------------------- Main -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Tải dữ liệu Inside Airbnb theo file config YAML")
    ap.add_argument("--config", help="Đường dẫn file yaml (mặc định: tự tìm trong configs/)")
    ap.add_argument("--force", action="store_true", help="Tải lại và ghi đè tệp đã có")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ in URL và nơi lưu, không tải")
    args = ap.parse_args()

    cfg_path = find_config(args.config).resolve()
    cfg = load_config(cfg_path)

    # raw_dir tính theo thư mục gốc dự án (cha của configs/) nên không phụ thuộc nơi chạy lệnh
    project_root = cfg_path.parent.parent
    out_root = Path(cfg["raw_dir"])
    if not out_root.is_absolute():
        out_root = project_root / out_root

    print(f"Config : {cfg_path}")
    print(f"Thành phố: {cfg.get('city', '?')} ({cfg.get('country', '?')}) | Nguồn: {cfg.get('source', '?')}")
    print(f"Lưu vào: {out_root}")

    jobs = build_jobs(cfg, out_root)
    stats = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0}

    last_folder = None
    for url, dest in jobs:
        if dest.parent != last_folder:
            last_folder = dest.parent
            print(f"\n=== {dest.parent.name} ===")
        if args.dry_run:
            print(f"  {url}\n    -> {dest}")
            continue
        stats[download_file(url, dest, args.force)] += 1

    if args.dry_run:
        print(f"\n(dry-run) Tổng cộng {len(jobs)} tệp.")
        return 0

    print("\n--- Tổng kết ---")
    print(f"Tải mới: {stats['ok']} | Đã có: {stats['skipped']} | "
          f"Không tìm thấy: {stats['missing']} | Lỗi: {stats['failed']}")
    if stats["missing"]:
        print("Gợi ý: kiểm tra lại ngày mốc trên https://insideairbnb.com/get-the-data/")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())