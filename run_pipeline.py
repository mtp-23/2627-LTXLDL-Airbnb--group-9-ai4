"""Chạy toàn bộ quy trình Inside Airbnb bằng một lệnh.

    python run_pipeline.py --config configs/santiago.yaml

Phân vai: mỗi mô-đun trong src/ chỉ làm đúng một việc và chỉ nhận/trả dữ liệu
trong bộ nhớ. Mọi việc ngoài nhiệm vụ gốc của từng mô-đun nằm ở đây:
đọc cấu hình, đường dẫn, đọc/ghi tệp, biến môi trường (khóa API), cache LLM,
chọn review sạch cho LLM/baseline, in kết quả và nối các bước theo thứ tự:

    download -> assign -> qa -> llm -> baseline -> evaluate -> kpi -> viz
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src import assign, baseline, download, evaluate, kpi, llm, qa, viz

ROOT = Path(__file__).resolve().parent
CHUNKSIZE = 100_000


# ======================================================================
# Cấu hình và đường dẫn
# ======================================================================
def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Cấu hình không hợp lệ: {path}")
    for key in ("city", "data_root", "raw_dir", "processed_dir", "download", "qa"):
        if not cfg.get(key):
            raise ValueError(f"Thiếu khóa '{key}' trong cấu hình")

    snaps = cfg.get("snapshots")
    if not isinstance(snaps, list) or not snaps:
        raise ValueError("Config phải có danh sách 'snapshots' không rỗng")
    cfg["snapshots"] = [str(s) for s in snaps]  # YAML có thể tự đổi ngày thành kiểu date
    if len(set(cfg["snapshots"])) != len(cfg["snapshots"]):
        raise ValueError("Các mốc trong 'snapshots' phải khác nhau")

    q = cfg["qa"]
    for k in ("sheet_id", "gid", "key_col", "room_types"):
        if not q.get(k):
            raise ValueError(f"Thiếu 'qa.{k}' trong cấu hình")
    if set(q["gid"]) != set(assign.SHEETS):
        raise ValueError("'qa.gid' phải chứa listing, review và area")
    if not all(isinstance(q.get(k), int) for k in ("area_min", "area_max")):
        raise ValueError("Thiếu 'qa.area_min'/'qa.area_max'")
    for k in ("number_of_reviews", "missing_price"):
        q[k] = {str(d): n for d, n in q[k].items()} if isinstance(q.get(k), dict) else {}
    q.setdefault("calendar_days", 365)

    cfg.setdefault("kpi", {}).setdefault("area_col", "neighbourhood_cleansed")
    cfg.setdefault("llm", {})
    return cfg


def resolve_config(value: str | None) -> Path:
    path = Path(value) if value else ROOT / "configs" / "santiago.yaml"
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    return path


def path_of(value: str | Path) -> Path:
    return ROOT / value  # giá trị tuyệt đối thì giữ nguyên


# ======================================================================
# Đọc / ghi tệp (nơi duy nhất trong dự án chạm vào đĩa dữ liệu)
# ======================================================================
def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def require_columns(df: pd.DataFrame, required, name) -> pd.DataFrame:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} thiếu các cột: {sorted(missing)}")
    return df


def read_csv(path: Path, required=()) -> pd.DataFrame:
    """Đọc CSV (có thể .gz) thành chuỗi, giữ nguyên ô rỗng là ''."""
    return require_columns(pd.read_csv(path, dtype=str, keep_default_na=False), required, path)


def read_csv_chunks(path: Path, usecols):
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols, chunksize=CHUNKSIZE)


def write_csv(df: pd.DataFrame, path: Path, encoding: str = "utf-8") -> Path:
    """Ghi CSV; chỉ nén gzip khi đường dẫn có đuôi .gz."""
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    _atomic(path, lambda t: df.to_csv(t, index=False, encoding=encoding, compression=compression))
    return path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    def write(tmp):
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            for r in records:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic(path, write)


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")


def dedupe_records(records: list[dict]) -> list[dict]:
    """Mỗi review_id một bản ghi; bản ghi sau giữ lại."""
    return list({str(r["review_id"]): r for r in records if "review_id" in r}.values())


# ======================================================================
# Các bước
# ======================================================================
def stage_download(cfg: dict, force: bool) -> None:
    stats = download.download_all(cfg["download"], cfg["data_root"], cfg["snapshots"],
                                  path_of(cfg["raw_dir"]), force)
    print(f"[download]\n mới={stats['done']}\n có sẵn={stats['skipped']}\n thiếu={stats['missing']}\n lỗi={stats['failed']}")
    if stats["missing"] or stats["failed"]:
        raise RuntimeError("Tải dữ liệu không hoàn tất")


def stage_assign(cfg: dict, sheet_dir: Path):
    """Nạp bảng phân công, cắt dữ liệu thô theo ID được giao, lưu vào raw_dir/phan_cong."""
    q, raw = cfg["qa"], path_of(cfg["raw_dir"])
    print("Đang phân công nhiệm vụ!")
    for name in assign.SHEETS:  # sheet chưa có thì tải về
        dest = sheet_dir / f"{name}.csv.gz"
        if not dest.is_file():
            content = download.fetch_sheet(q["sheet_id"], q["gid"][name])
            _atomic(dest, lambda t: t.write_bytes(gzip.compress(content)))
    needed = assign.required_sheet_columns(q)
    listing_a, review_a = assign.parse_assignments(
        read_csv(sheet_dir / "listing.csv.gz", needed["listing"]),
        read_csv(sheet_dir / "review.csv.gz", needed["review"]), q)

    assigned, city_reviews = {}, 0
    latest_snap = max(cfg["snapshots"])
    for s in cfg["snapshots"]:
        raw_reviews = read_csv(raw / s / "reviews.csv.gz", assign.RAW_REQUIRED["reviews"])
        if s == latest_snap:
            city_reviews = len(raw_reviews)  # mốc mới nhất quyết định số review toàn thành phố
        assigned[s] = {
            "listings": assign.subset_listings(
                read_csv(raw / s / "listings.csv.gz", assign.RAW_REQUIRED["listings"]), listing_a),
            "calendar": assign.subset_calendar(
                read_csv_chunks(raw / s / "calendar.csv.gz", assign.CALENDAR_COLUMNS), listing_a, q["calendar_days"]),
            "reviews": assign.subset_reviews(raw_reviews, review_a),
        }
        for table, df in assigned[s].items():
            write_csv(df, sheet_dir / s / f"{table}.csv.gz")
    return assigned, review_a, city_reviews


def stage_qa(cfg: dict, assigned: dict, sheet_dir: Path) -> dict:
    for s, tables in assigned.items():
        for table, df in tables.items():
            require_columns(df, qa.REQUIRED[table] | set(qa.TABLES[table]), f"{table} {s}")
    area = read_csv(sheet_dir / "area.csv.gz", qa.AREA_SHEET_REQUIRED)
    bounds = qa.parse_area_bounds(area, cfg["city"])
    flags = qa.run_qa(assigned, bounds, cfg["qa"], cfg["qa"]["calendar_days"], log=print)
    flagged_dir = path_of(cfg["processed_dir"]) / "flagged"
    for (s, table), df in flags.items():
        write_csv(df, flagged_dir / s / f"{table}_flags.csv.gz")
    return flags


def clean_reviews(assigned: dict, flags: dict, snapshots: list[str]) -> pd.DataFrame:
    """Review sạch sau QA: bỏ ID bị cờ missing_comments ở bất kỳ mốc nào; nội dung mốc mới nhất thắng."""
    frames, bad = [], set()
    for s in snapshots:
        frames.append(assigned[s]["reviews"][["id", "comments"]])
        f = flags[s, "reviews"]
        bad |= set(f.loc[f["missing_comments"], "id"])
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["id"].isin(bad) & df["comments"].str.strip().ne("")]
    return df.drop_duplicates("id", keep="last").reset_index(drop=True)


def stage_llm(cfg: dict, clean: pd.DataFrame, sample_ids: set[str], city_reviews: int, skip: bool) -> list[dict]:
    """Chạy LLM trên mẫu được giao (có cache) và trả về kết quả trong mẫu đã khử trùng."""
    conf = cfg["llm"]
    cache = path_of(conf.get("cache", "data/processed/llm_cache.jsonl"))
    out_csv = path_of(cfg["processed_dir"]) / "llm_results.csv"
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")
    batch_size = conf.get("batch_size") or 100

    if skip:
        print("[llm] --skip-llm: không gọi Gemini, dùng cache hiện có")
    elif not api_key:
        print("[llm] thiếu GEMINI_API_KEY, không gọi Gemini; dùng cache hiện có")
    else:
        target = clean[clean["id"].isin(sample_ids)]
        cached = {str(r["review_id"]) for r in read_jsonl(cache)}
        pending = target[~target["id"].isin(cached)]
        print(f"[llm]\n mục tiêu: {len(target)}/{len(sample_ids)}\n đã cache: {len(cached & set(target['id']))}\n cần gửi LLM: {len(pending)}")
        if not pending.empty:
            asyncio.run(llm.analyze_reviews(
                pending, api_key, conf.get("model", "gemini-3.5-flash-lite"),
                batch_size=batch_size, aspects=conf.get("aspects"),
                on_batch=lambda res: append_jsonl(cache, res)))

    cost = llm.estimate_total_cost(city_reviews, batch_size=batch_size)
    print(f"[llm] Ước tính chi phí nếu xử lý toàn bộ {city_reviews:,} review thành phố: {cost}")

    records = dedupe_records(read_jsonl(cache))
    in_sample = [r for r in records if str(r["review_id"]) in sample_ids]
    written = write_csv(llm.flatten_results(in_sample), out_csv)
    print(f"[llm] xuất {len(in_sample)} mẫu LLM ra {written.relative_to(ROOT)}")
    return in_sample


def stage_baseline(cfg: dict, clean: pd.DataFrame) -> list[dict]:
    output = path_of(cfg["llm"].get("baseline_output", "data/processed/baseline_results.jsonl"))
    if output.suffix.lower() != ".jsonl":
        raise ValueError("llm.baseline_output phải là đường dẫn .jsonl")
    records = baseline.analyze(dict(zip(clean["id"], clean["comments"])), cfg["llm"]["aspects"])
    write_jsonl(records, output)
    print(f"[baseline] {len(records)} review -> {output.relative_to(ROOT)}")
    return records


def stage_evaluate(cfg: dict, llm_records: list[dict], baseline_records: list[dict]) -> dict | None:
    if not llm_records:
        print("[evaluate] chưa có kết quả LLM: bỏ qua đánh giá (KPI 7, 8 sẽ 'chưa có')")
        return None
    processed = path_of(cfg["processed_dir"])
    truth_path = path_of(cfg.get("evaluation", {}).get("ground_truth", "data/samples/hand_labeled.csv"))
    truth = read_csv(truth_path, {"review_id"})
    result = evaluate.evaluate(truth, pd.DataFrame(llm_records), pd.DataFrame(baseline_records),
                               cfg["llm"]["aspects"], llm.SENTIMENTS)
    print(f"[evaluate] chấm {result['n']} review có nhãn tay đầy đủ (trên {result['n_truth']} nhãn tay)")
    show = ["Method", "Sentiment Acc", "Sentiment Macro-F1", "Aspect Micro-F1", "Aspect Macro-F1", "Language Acc"]
    print(result["metrics"][show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    write_csv(result["metrics"], processed / "evaluation_metrics.csv")
    write_csv(result["errors"], processed / "llm_error_analysis.csv")
    return result


def stage_kpi(cfg: dict, assigned: dict, flags: dict, llm_records: list[dict],
              evaluation: dict | None, snapshot: str | None) -> dict:
    """Tính 8 KPI trước/sau QA, ghi kpi_*.csv và dữ liệu cho hình; trả về bảng cho viz."""
    processed = path_of(cfg["processed_dir"])
    area_col = cfg["kpi"]["area_col"]
    currency = cfg.get("currency", "CLP")
    snap = snapshot or max(cfg["snapshots"])

    after_data = {s: kpi.clean_after_qa(assigned[s]["listings"], assigned[s]["calendar"],
                                        flags[s, "listings"], flags[s, "calendar"])
                  for s in cfg["snapshots"]}
    l_after, c_after = after_data[snap]

    llm_before = llm_after = None
    if llm_records:
        llm_before = llm.flatten_results(llm_records)
        llm_before["sentiment"] = llm_before["sentiment"].astype(str).str.strip().str.lower()
        llm_after = llm_before[llm_before["sentiment"].isin(llm.SENTIMENTS)]
    llm_f1 = (evaluation["llm_sentiment_f1"], evaluation["n"]) if evaluation else None

    before_by_snap = {
        s: kpi.compute_kpis(assigned[s]["listings"], assigned[s]["calendar"], area_col, qa.ROOM_TYPES, llm_before, llm_f1)
        for s in cfg["snapshots"]
    }
    after_by_snap = {
        s: kpi.compute_kpis(after_data[s][0], after_data[s][1], area_col, qa.ROOM_TYPES, llm_after, llm_f1, flags[s, "listings"])
        for s in cfg["snapshots"]
    }
    before, after = before_by_snap[snap], after_by_snap[snap]

    print(kpi.render_table(after, [
        f"KẾT QUẢ 8 KPI SAU QA - {cfg['city']} - snapshot {snap}",
        f"Sau QA: {len(l_after):,} listing, {len(c_after):,} ngày lịch. KPI 7, 8: mẫu review đã chạy LLM"], currency))

    path_before = write_csv(kpi.kpi_table(before, snap, currency), processed / "kpi_before_qa.csv", "utf-8-sig")
    path_after = write_csv(kpi.kpi_table(after, snap, currency), processed / "kpi_after_qa.csv", "utf-8-sig")
    qa_report_path = write_csv(qa.build_qa_report(flags, before_by_snap, after_by_snap, currency),
                               ROOT / "reports" / "qa_report.csv", "utf-8-sig")
    print(f"\nĐã ghi KPI trước QA : {path_before.relative_to(ROOT)}")
    print(f"Đã ghi KPI sau QA   : {path_after.relative_to(ROOT)}")
    print(f"Đã ghi báo cáo QA   : {qa_report_path.relative_to(ROOT)}")

    fig1 = kpi.kpi_by_area(l_after, c_after, snap, area_col)
    fig2 = kpi.kpi_over_time(after_data, area_col, qa.ROOM_TYPES)
    fig3 = evaluation["sentiment_report"] if evaluation else None
    viz_dir = processed / "viz"
    write_csv(fig1, viz_dir / "fig1_areas.csv", "utf-8-sig")
    write_csv(fig2, viz_dir / "fig2_over_time.csv", "utf-8-sig")
    if fig3 is not None:
        write_csv(fig3, viz_dir / "fig3_llm.csv", "utf-8-sig")
    else:
        (viz_dir / "fig3_llm.csv").unlink(missing_ok=True)  # tránh vẽ lại số liệu LLM cũ
    return {"fig1": fig1, "fig2": fig2, "fig3": fig3, "snapshot": snap}


def stage_viz(cfg: dict, tables: dict) -> None:
    geojson = path_of(cfg["raw_dir"]) / tables["snapshot"] / "neighbourhoods.geojson"
    features = None
    if geojson.is_file():
        with geojson.open(encoding="utf-8") as stream:
            features = json.load(stream)["features"]
    out_dir = ROOT / "figures"
    drawn = viz.draw_all(tables["fig1"], tables["fig2"], tables["fig3"], features, out_dir, llm.SENTIMENTS)
    print(f"[viz] hình trong {out_dir.relative_to(ROOT)}/")
    reasons = {"fig1": f"thiếu {geojson.name} của mốc {tables['snapshot']} (hãy bật tải GeoJSON)",
               "fig2": "thiếu dữ liệu", "fig3": "chưa có kết quả LLM"}
    for name, path in drawn.items():
        print(f"  ghi {path.name}" if path else f"  [bỏ qua] {name}: {reasons[name]}")


# ======================================================================
# Điểm vào
# ======================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline phân tích Inside Airbnb")
    parser.add_argument("--config", help="File cấu hình YAML, mặc định configs/santiago.yaml")
    parser.add_argument("--force", action="store_true", help="Tải lại dữ liệu đã có")
    parser.add_argument("--skip-download", action="store_true", help="Dùng dữ liệu thô đã tải")
    parser.add_argument("--skip-llm", action="store_true", help="Không gọi Gemini, dùng cache hiện có")
    parser.add_argument("--skip-evaluation", action="store_true", help="Không đánh giá LLM với nhãn tay")
    parser.add_argument("--snapshot", help="Snapshot tính KPI 1-6, mặc định là snapshot mới nhất")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        cfg = load_config(resolve_config(args.config))
        if args.snapshot and args.snapshot not in cfg["snapshots"]:
            raise ValueError(f"Snapshot không có trong config: {args.snapshot}")
        sheet_dir = path_of(cfg["raw_dir"]) / "phan_cong"

        if not args.skip_download:
            stage_download(cfg, args.force)
        assigned, review_a, city_reviews = stage_assign(cfg, sheet_dir)
        flags = stage_qa(cfg, assigned, sheet_dir)

        clean = clean_reviews(assigned, flags, cfg["snapshots"])
        llm_records = stage_llm(cfg, clean, assign.llm_sample_ids(review_a), city_reviews, args.skip_llm)
        baseline_records = stage_baseline(cfg, clean)
        evaluation = None if args.skip_evaluation else stage_evaluate(cfg, llm_records, baseline_records)

        tables = stage_kpi(cfg, assigned, flags, llm_records, evaluation, args.snapshot)
        stage_viz(cfg, tables)
        print("Đã xong!")
        return 0
    except Exception as error:
        print(f"Pipeline thất bại: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
