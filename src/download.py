import argparse
import csv
import gzip
import json
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parent.parent
FILE_KEYS = {
    "listings": {"id"},
    "calendar": {"listing_id", "date"},
    "reviews": {"id"},
}


def valid_csv(path: Path, file_type: str) -> bool:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        columns = set(next(csv.reader(file)))
    return FILE_KEYS[file_type] <= columns


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "santiago.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = ROOT / config["raw_dir"]
    download_config = config["download"]

    for snapshot in config["snapshots"]:
        snapshot = str(snapshot)
        snapshot_dir = raw_dir / snapshot

        for file_type in download_config["f_type"]:
            output_path = snapshot_dir / f"{file_type}.csv.gz"
            if output_path.is_file() and not args.force and valid_csv(output_path, file_type):
                print(f"Đã có và hợp lệ: {output_path}")
                continue

            url = download_config["URL_CSV"].format(
                data_root=config["data_root"].strip("/"),
                date=snapshot,
                f_type=file_type,
            )
            download(url, output_path)
            if not valid_csv(output_path, file_type):
                raise ValueError(f"Tệp CSV thiếu cột khóa: {output_path}")
            print(f"Đã tải và kiểm tra: {output_path}")

        geojson_path = snapshot_dir / "neighbourhoods.geojson"
        if not geojson_path.is_file() or args.force:
            geojson_url = download_config["URL_GEOJSON"].format(
                data_root=config["data_root"].strip("/"),
                date=snapshot,
            )
            download(geojson_url, geojson_path)

        with geojson_path.open(encoding="utf-8") as file:
            geojson = json.load(file)
        if geojson["type"] != "FeatureCollection" or not isinstance(geojson["features"], list):
            raise ValueError(f"GeoJSON không hợp lệ: {geojson_path}")
        print(f"Đã có GeoJSON hợp lệ: {geojson_path}")


if __name__ == "__main__":
    main()
