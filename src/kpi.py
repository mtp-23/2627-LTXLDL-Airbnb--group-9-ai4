import argparse
import json
import shutil
import sys
import textwrap
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

AREA_COL = "neighbourhood_cleansed"
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
SENTIMENTS = ["positive", "mixed", "negative"]

# Cờ QA làm dòng bị LOẠI khỏi dữ liệu sạch (tên khớp với RULES trong qa.py)
EXCLUDE_FLAGS = {
    "listings": ["missing_neigh_cleansed", "missing_room_type", "missing_number_of_reviews",
                 "invalid_reviews_num", "invalid_room_type"],
    "calendar": ["invalid_calendar_date", "invalid_calendar_available"],
}
KEY_COLUMNS = {"listings": ["id"], "calendar": ["listing_id", "date"]}
NEEDED_COLUMNS = {"listings": ["id", AREA_COL, "room_type", "number_of_reviews"],
           "calendar": ["listing_id", "date", "available"]}

# 7 KPI đúng tên, công thức, đơn vị trong bảng KPI của nhóm
KPI = {
    "KPI1": {"number": 1, "name": "Số lượng listing theo khu vực", "formula": "Đếm listing theo neighbourhood_cleansed",
             "unit": "Phòng", "decimals": 0},
    "KPI2": {"number": 2, "name": "Tỷ lệ loại phòng", "formula": "Số listing từng room_type / tổng listing × 100%",
             "unit": "%", "decimals": 2},
    "KPI3": {"number": 3, "name": "Tỷ lệ listing chưa có review",
             "formula": "Listing có number_of_reviews = 0 / tổng listing × 100%", "unit": "%", "decimals": 2},
    "KPI4": {"number": 4, "name": "Số review trung bình trên listing", "formula": "Tổng number_of_reviews / tổng listing",
             "unit": "Review/listing", "decimals": 2},
    "KPI5": {"number": 5, "name": "Tỷ lệ ngày có thể đặt phòng",
             "formula": "Số ngày available = 't' / tổng ngày hợp lệ × 100%", "unit": "%", "decimals": 2},
    "KPI6": {"number": 6, "name": "Tỷ lệ review tích cực theo LLM",
             "formula": "Review được LLM phân loại tích cực / tổng review có nhãn hợp lệ × 100%", "unit": "%", "decimals": 2},
    "KPI7": {"number": 7, "name": "F1-score của LLM", "formula": "So sánh nhãn LLM với nhãn thủ công",
             "unit": "Điểm F1 (0-1)", "decimals": 4},
}

# 1. ĐỌC DỮ LIỆU DO qa.py TẠO RA
def read_config(path):
    path = Path(path) if path else sorted((ROOT / "configs").glob("*.yaml"))[0]
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["snapshots"] = [str(d) for d in cfg["snapshots"]]
    return cfg

def resolve_dir(cfg, key, default):
    p = Path(cfg.get(key) or default)
    return p if p.is_absolute() else ROOT / p

def read_before_qa(raw_dir, snap):
    """Dữ liệu trước QA của một snapshot: (listings, calendar) lấy từ data/raw/phan_cong/<ngày>/."""
    results = []
    for table in ("listings", "calendar"):
        path = raw_dir / "phan_cong" / snap / f"{table}.csv.gz"
        if not path.exists():
            raise SystemExit(f"Chưa có {path.relative_to(ROOT)}.\nHãy chạy qa.py trước (python src/qa.py).")
        columns = NEEDED_COLUMNS[table]
        df = pd.read_csv(path, dtype=str, usecols=lambda c: c in columns)
        for c in columns:
            if c not in df.columns:
                df[c] = pd.NA
        results.append(df[columns].reset_index(drop=True))
    return tuple(results)

def rows_flagged_for_exclusion(proc, snap, table, df):
    """True với dòng bị gắn cờ 'loại bỏ' trong file cờ do qa.py tạo."""
    path = proc / "flagged" / snap / f"{table}_flags.csv.gz"
    if not path.exists():
        raise SystemExit(f"Chưa có {path.relative_to(ROOT)}.\nHãy chạy qa.py trước (python src/qa.py).")
    flags = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = [c for c in EXCLUDE_FLAGS[table] if c in flags.columns]
    for c in columns:
        flags[c] = flags[c].str.lower().eq("true")
    flags = flags[KEY_COLUMNS[table] + columns].drop_duplicates(KEY_COLUMNS[table])
    merged_keys = df[KEY_COLUMNS[table]].astype(str).merge(flags, on=KEY_COLUMNS[table], how="left")
    return pd.Series(merged_keys[columns].fillna(False).any(axis=1).values, index=df.index)

def clean_after_qa(proc, snap, listings, calendar):
    """Sau QA: liên kết dữ liệu phan_cong với file cờ, bỏ dòng bị cờ loại bỏ, dòng trùng khóa và lịch của listing đã bị loại."""
    l = listings[~(rows_flagged_for_exclusion(proc, snap, "listings", listings) | listings.duplicated("id"))]
    c = calendar[~(rows_flagged_for_exclusion(proc, snap, "calendar", calendar) | calendar.duplicated(["listing_id", "date"]))]
    c = c[c["listing_id"].isin(set(l["id"]))]
    return l.reset_index(drop=True), c.reset_index(drop=True)

def read_llm_results(path, keep_valid_labels_only):
    """Đọc llm_cache.jsonl (mỗi dòng có review_id và overall_sentiment). Sau QA chỉ giữ nhãn hợp lệ."""
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for json_line in f:
            if json_line.strip():
                it = json.loads(json_line)
                rows.append({"review_id": str(it.get("review_id")),
                             "llm_label": str(it.get("overall_sentiment")).strip().lower()})
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("review_id", keep="last")
    return df[df["llm_label"].isin(SENTIMENTS)] if keep_valid_labels_only else df

def read_hand_labels(keep_valid_labels_only):
    path = ROOT / "data/samples/hand_labeled.csv"
    if not path.exists():
        return None
    hand = pd.read_csv(path, dtype=str, encoding="utf-8-sig")[["review_id", "label"]].dropna()
    hand["label"] = hand["label"].str.strip().str.lower()
    return hand[hand["label"].isin(SENTIMENTS)] if keep_valid_labels_only else hand

# 2. TÍNH 7 KPI: mỗi KPI trả về (giá trị, tên nhóm, tử số, mẫu số)
def pct(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else None


def compute_kpi_1_to_5(l, c):
    result = {}
    # KPI1: đếm listing theo neighbourhood_cleansed -> một giá trị: số listing của khu vực lớn nhất (kèm tên)
    by_area = l.dropna(subset=[AREA_COL]).groupby(AREA_COL).size()
    if len(by_area):
        result["KPI1"] = (int(by_area.max()), by_area.idxmax(), int(by_area.max()), int(by_area.sum()))
    else:
        result["KPI1"] = (None, "", None, 0)

    # KPI2: số listing từng room_type / tổng listing -> một giá trị: tỷ lệ của loại phòng lớn nhất (kèm tên)
    counts = {rt: int((l["room_type"] == rt).sum()) for rt in ROOM_TYPES}
    top = max(counts, key=counts.get)
    result["KPI2"] = (pct(counts[top], len(l)), top, counts[top], len(l))

    # KPI3: tỷ lệ listing chưa có review; KPI4: review trung bình trên listing
    review_counts = pd.to_numeric(l["number_of_reviews"], errors="coerce")
    without_reviews = int((review_counts == 0).sum())
    result["KPI3"] = (pct(without_reviews, len(l)), "", without_reviews, len(l))
    total = float(review_counts.sum())
    result["KPI4"] = (round(total / len(l), 2) if len(l) else None, "", total, len(l))

    # KPI5: tỷ lệ ngày có available = 't'
    available_days = int(c["available"].astype("string").str.strip().str.lower().eq("t").sum())
    result["KPI5"] = (pct(available_days, len(c)), "", available_days, len(c))
    return result


def f1_scores(hand_labels, llm_labels, classes):
    """precision/recall/F1 từng lớp, macro F1, accuracy và ma trận nhầm lẫn (hàng = nhãn tay, cột = LLM)."""
    cm = pd.crosstab(pd.Series(list(hand_labels), name="t"), pd.Series(list(llm_labels), name="p"))
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)
    per_class, class_f1_values = {}, []
    for k in classes:
        tp = int(cm.loc[k, k])
        fp, fn = int(cm[k].sum()) - tp, int(cm.loc[k].sum()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[k] = (round(p, 4), round(r, 4), round(f1, 4), tp + fn)
        if tp + fn:
            class_f1_values.append(f1)                  # macro F1 chỉ tính lớp có xuất hiện trong nhãn tay
    total = int(cm.values.sum())
    acc = sum(int(cm.loc[k, k]) for k in classes) / total if total else None
    macro = sum(class_f1_values) / len(class_f1_values) if class_f1_values else None
    return per_class, macro, acc, cm

def compute_kpi_6_7(llm, hand):
    result = {}
    if llm is None or llm.empty:
        return result
    # KPI6: tỷ lệ review tích cực theo LLM
    positive_count = int((llm["llm_label"] == "positive").sum())
    result["KPI6"] = (pct(positive_count, len(llm)), "", positive_count, len(llm))
    # KPI7: macro F1 so với nhãn tay (chỉ review có cả hai nhãn)
    if hand is not None:
        merged = hand.merge(llm, on="review_id")
        if len(merged):
            macro = f1_scores(merged["label"], merged["llm_label"], sorted(set(merged["label"]) | set(merged["llm_label"])))[1]
            result["KPI7"] = (None if macro is None else round(macro, 4), "", None, len(merged))
    return result

# 3. DỮ LIỆU CHO viz.py (tính sẵn ở đây, viz.py chỉ vẽ)
def build_fig1_data(l, c, snap):
    """Hình 1 (bản đồ): số listing và tỷ lệ ngày còn trống theo từng khu vực, sau QA."""
    listing_counts = l.dropna(subset=[AREA_COL]).groupby(AREA_COL).size().rename("listing_count")
    is_available = c["available"].astype("string").str.strip().str.lower().eq("t")
    area_lookup = l.dropna(subset=["id", AREA_COL]).drop_duplicates("id")[["id", AREA_COL]]
    calendar_stats = (c.assign(is_available_day=is_available).merge(area_lookup, left_on="listing_id", right_on="id")
            .groupby(AREA_COL)["is_available_day"].agg(["sum", "count"]))
    calendar_stats.columns = ["available_days", "total_days"]
    result = pd.concat([listing_counts, calendar_stats], axis=1).reset_index().rename(columns={AREA_COL: "area"})
    result["available_day_rate_pct"] = [pct(a, b) if b else None for a, b in zip(result["available_days"], result["total_days"])]
    result.insert(0, "snapshot", snap)
    return result

def build_fig2_data(data_after_qa):
    """Hình 2 (theo thời gian): KPI 3, 4, 5 của từng snapshot, sau QA."""
    rows = []
    for snap, (l, c) in data_after_qa.items():
        result = compute_kpi_1_to_5(l, c)
        rows.append({"snapshot": snap, "no_review_rate_pct": result["KPI3"][0],
                     "avg_reviews_per_listing": result["KPI4"][0], "available_day_rate_pct": result["KPI5"][0]})
    return pd.DataFrame(rows)

def build_fig3_data(llm_after, hand_after):
    """Hình 3 (LLM): ma trận nhầm lẫn và F1 từng lớp của LLM so với nhãn tay, sau QA. None nếu chưa có kết quả."""
    if llm_after is None or hand_after is None:
        return None
    merged = hand_after.merge(llm_after, on="review_id")
    if merged.empty:
        return None
    per_class, macro, acc, cm = f1_scores(merged["label"], merged["llm_label"], SENTIMENTS)
    rows = [{"kind": "confusion", "true_label": t, "llm_label": p, "metric": "review_count", "value": int(cm.loc[t, p]),
             "sample_size": len(merged)} for t in SENTIMENTS for p in SENTIMENTS]
    for k, (p, r, f1, n) in per_class.items():
        rows += [{"kind": "class", "true_label": k, "llm_label": "", "metric": metric_name, "value": v, "sample_size": n}
                 for metric_name, v in (("precision", p), ("recall", r), ("f1", f1))]
    rows += [{"kind": "total", "true_label": "", "llm_label": "", "metric": "macro_f1", "value": macro, "sample_size": len(merged)},
             {"kind": "total", "true_label": "", "llm_label": "", "metric": "accuracy", "value": acc, "sample_size": len(merged)}]
    return pd.DataFrame(rows)

# 4. IN BẢNG RA TERMINAL VÀ GHI CSV
def format_number(v, decimals):
    if v is None or pd.isna(v):
        return "chưa có"
    return f"{v:,.0f}" if decimals == 0 else f"{v:,.{decimals}f}"

def calculation_basis(kid, result):
    """Cột 'Cơ sở tính': tử số / mẫu số (và tên nhóm với KPI1, KPI2) để biết con số lấy từ đâu."""
    _, detail, numerator, denominator = result.get(kid, (None, "", None, None))
    if denominator is None:
        return ""
    unit_label = "ngày" if kid == "KPI5" else ("review" if kid in ("KPI6", "KPI7") else "listing")
    if kid == "KPI1":
        return f"{detail} (lớn nhất) / {denominator:,} {unit_label}"
    if kid == "KPI2":
        return f"{detail} (lớn nhất): {numerator:,} / {denominator:,} {unit_label}"
    if kid == "KPI7":
        return f"{denominator:,} review có đủ 2 nhãn"
    return f"{numerator:,.0f} / {denominator:,} {unit_label}"

def print_table(title_lines, headers, rows, right_aligned, stretch_columns):
    """
    In bảng căn lề, chia cột. Khung ASCII để terminal nào cũng hiển thị đúng.
    Bảng tự co theo bề rộng terminal: các cột trong `cot_co_gian` được xuống dòng khi quá dài.
    """
    normalize_text = lambda x: unicodedata.normalize("NFC", str(x))
    headers = [normalize_text(h) for h in headers]
    rows = [[normalize_text(x) for x in r] for r in rows]
    natural_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    available_width = shutil.get_terminal_size((100, 24)).columns - (3 * len(headers) + 1) - 1
    widths = list(natural_widths)
    if sum(widths) > available_width:
        fixed_width = sum(w for i, w in enumerate(natural_widths) if i not in stretch_columns)
        remaining_width = max(available_width - fixed_width, 18 * len(stretch_columns))
        stretch_total = sum(natural_widths[i] for i in stretch_columns)
        for i in stretch_columns:
            widths[i] = max(14, int(remaining_width * natural_widths[i] / stretch_total))

    def horizontal_line():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def print_row(cells):
        wrapped_cells = [textwrap.wrap(c, width=w) or [""] for c, w in zip(cells, widths)]
        for k in range(max(len(x) for x in wrapped_cells)):
            out = []
            for i, (cell_lines, w) in enumerate(zip(wrapped_cells, widths)):
                c = cell_lines[k] if k < len(cell_lines) else ""
                out.append(c.rjust(w) if i in right_aligned else c.ljust(w))
            print("| " + " | ".join(out) + " |")

    print()
    for line in title_lines:
        print(line)
    print(horizontal_line())
    print_row(headers)
    print(horizontal_line())
    for r in rows:
        print_row(r)
        print(horizontal_line())

def kpi_csv_table(result, snap):
    """File KPI: đúng 7 dòng, mỗi KPI một giá trị (KPI chưa tính được để trống)."""
    rows = []
    for kid, meta in KPI.items():
        value, group, numerator, denominator = result.get(kid, (None, "", None, None))
        rows.append({"number": meta["number"], "kpi": meta["name"], "formula": meta["formula"], "unit": meta["unit"],
                     "value": value, "group": group, "numerator": numerator, "denominator": denominator,
                     "scope": snap if kid in ("KPI1", "KPI2", "KPI3", "KPI4", "KPI5") else "mẫu review đã chạy LLM"})
    return pd.DataFrame(rows)

def run(config_path=None, snapshot=None, skip_llm=False, llm_cache=None):
    try:
        sys.stdout.reconfigure(errors="replace")       # tránh lỗi mã hóa trên một số terminal Windows
    except (AttributeError, ValueError):
        pass
    cfg = read_config(config_path)
    raw_dir, proc = resolve_dir(cfg, "raw_dir", "data/raw"), resolve_dir(cfg, "processed_dir", "data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    snap = str(snapshot) if snapshot else sorted(cfg["snapshots"])[-1]
    if snap not in cfg["snapshots"]:
        raise SystemExit(f"Snapshot {snap} không có trong config. Các mốc hợp lệ: {', '.join(cfg['snapshots'])}")

    # ---- trước QA: chỉ cần data/raw/phan_cong của snapshot được chọn ----
    listings_before, calendar_before = read_before_qa(raw_dir, snap)
    # ---- sau QA: phan_cong liên kết với flagged; cần đủ các snapshot cho hình theo thời gian ----
    data_after_qa = {}
    for s in cfg["snapshots"]:
        l, c = (listings_before, calendar_before) if s == snap else read_before_qa(raw_dir, s)
        data_after_qa[s] = clean_after_qa(proc, s, l, c)
    listings_after, calendar_after = data_after_qa[snap]
    before, after = compute_kpi_1_to_5(listings_before, calendar_before), compute_kpi_1_to_5(listings_after, calendar_after)

    # ---- KPI 6, 7 ----
    llm_after = hand_after = None
    if skip_llm:
        print("(--skip-llm: bỏ KPI 6, 7)")
    else:
        cache = Path(llm_cache) if llm_cache else proc / "llm_cache.jsonl"
        cache = cache if cache.is_absolute() else ROOT / cache
        llm_before, llm_after = read_llm_results(cache, False), read_llm_results(cache, True)
        hand_after = read_hand_labels(True)
        if llm_before is None:
            print(f"(chưa có {cache.name}: KPI 6, 7 hiện 'chưa có'; chạy bước LLM trước hoặc dùng --llm-cache)")
        before.update(compute_kpi_6_7(llm_before, read_hand_labels(False)))
        after.update(compute_kpi_6_7(llm_after, hand_after))

    # ---- bảng terminal: kết quả SAU QA ----
    rows = [[meta["number"], meta["name"], format_number(after.get(kid, (None,))[0], meta["decimals"]), meta["unit"],
             calculation_basis(kid, after)] for kid, meta in KPI.items()]
    print_table([f"KẾT QUẢ 7 KPI SAU QA - {cfg.get('city', '')} - snapshot {snap}",
             f"Sau QA: {len(listings_after):,} listing, {len(calendar_after):,} ngày lịch. KPI 6, 7: mẫu review đã chạy LLM"],
            ["#", "KPI", "Giá trị", "Đơn vị", "Cơ sở tính"], rows, right_aligned={0, 2}, stretch_columns=[1, 4])

    # ---- hai file KPI: trước QA và sau QA (mỗi file 7 dòng) ----
    path_before, path_after = proc / "kpi_before_qa.csv", proc / "kpi_after_qa.csv"
    kpi_csv_table(before, snap).to_csv(path_before, index=False, encoding="utf-8-sig")
    kpi_csv_table(after, snap).to_csv(path_after, index=False, encoding="utf-8-sig")

    # ---- dữ liệu cho viz.py ----
    viz_dir = proc / "viz"
    viz_dir.mkdir(exist_ok=True)
    build_fig1_data(listings_after, calendar_after, snap).to_csv(viz_dir / "fig1_areas.csv", index=False, encoding="utf-8-sig")
    build_fig2_data(data_after_qa).to_csv(viz_dir / "fig2_over_time.csv", index=False, encoding="utf-8-sig")
    fig3 = build_fig3_data(llm_after, hand_after)
    if fig3 is not None:
        fig3.to_csv(viz_dir / "fig3_llm.csv", index=False, encoding="utf-8-sig")
    elif (viz_dir / "fig3_llm.csv").exists():
        (viz_dir / "fig3_llm.csv").unlink()               # tránh vẽ lại số liệu LLM cũ khi chưa có kết quả mới

    print(f"\nĐã ghi KPI trước QA : {path_before.relative_to(ROOT)}")
    print(f"Đã ghi KPI sau QA   : {path_after.relative_to(ROOT)}")
    print(f"Dữ liệu cho viz.py  : {viz_dir.relative_to(ROOT)}")
    return kpi_csv_table(before, snap), kpi_csv_table(after, snap)

def main():
    ap = argparse.ArgumentParser(description="Tính 7 KPI trước QA và sau QA; in bảng sau QA ra terminal")
    ap.add_argument("--config")
    ap.add_argument("--snapshot", help="mốc tính KPI 1-5 (mặc định: mốc mới nhất trong config)")
    ap.add_argument("--skip-llm", action="store_true", help="bỏ KPI 6, 7")
    ap.add_argument("--llm-cache", help="file kết quả LLM (mặc định data/processed/llm_cache.jsonl)")
    a = ap.parse_args()
    run(a.config, a.snapshot, a.skip_llm, a.llm_cache)

if __name__ == "__main__":
    main()