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
CO_LOAI_BO = {
    "listings": ["missing_neigh_cleansed", "missing_room_type", "missing_number_of_reviews",
                 "invalid_reviews_num", "invalid_room_type"],
    "calendar": ["invalid_calendar_date", "invalid_calendar_available"],
}
KHOA = {"listings": ["id"], "calendar": ["listing_id", "date"]}
COT_CAN = {"listings": ["id", AREA_COL, "room_type", "number_of_reviews"],
           "calendar": ["listing_id", "date", "available"]}

# 7 KPI đúng tên, công thức, đơn vị trong bảng KPI của nhóm
KPI = {
    "KPI1": {"stt": 1, "ten": "Số lượng listing theo khu vực", "cong_thuc": "Đếm listing theo neighbourhood_cleansed",
             "don_vi": "Phòng", "so_le": 0},
    "KPI2": {"stt": 2, "ten": "Tỷ lệ loại phòng", "cong_thuc": "Số listing từng room_type / tổng listing × 100%",
             "don_vi": "%", "so_le": 2},
    "KPI3": {"stt": 3, "ten": "Tỷ lệ listing chưa có review",
             "cong_thuc": "Listing có number_of_reviews = 0 / tổng listing × 100%", "don_vi": "%", "so_le": 2},
    "KPI4": {"stt": 4, "ten": "Số review trung bình trên listing", "cong_thuc": "Tổng number_of_reviews / tổng listing",
             "don_vi": "Review/listing", "so_le": 2},
    "KPI5": {"stt": 5, "ten": "Tỷ lệ ngày có thể đặt phòng",
             "cong_thuc": "Số ngày available = 't' / tổng ngày hợp lệ × 100%", "don_vi": "%", "so_le": 2},
    "KPI6": {"stt": 6, "ten": "Tỷ lệ review tích cực theo LLM",
             "cong_thuc": "Review được LLM phân loại tích cực / tổng review có nhãn hợp lệ × 100%", "don_vi": "%", "so_le": 2},
    "KPI7": {"stt": 7, "ten": "F1-score của LLM", "cong_thuc": "So sánh nhãn LLM với nhãn thủ công",
             "don_vi": "Điểm F1 (0-1)", "so_le": 4},
}

# 1. ĐỌC DỮ LIỆU DO qa.py TẠO RA
def doc_config(path):
    path = Path(path) if path else sorted((ROOT / "configs").glob("*.yaml"))[0]
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["snapshots"] = [str(d) for d in cfg["snapshots"]]
    return cfg

def thu_muc(cfg, khoa, mac_dinh):
    p = Path(cfg.get(khoa) or mac_dinh)
    return p if p.is_absolute() else ROOT / p

def doc_truoc_qa(raw_dir, snap):
    """Dữ liệu trước QA của một snapshot: (listings, calendar) lấy từ data/raw/phan_cong/<ngày>/."""
    ket_qua = []
    for bang in ("listings", "calendar"):
        path = raw_dir / "phan_cong" / snap / f"{bang}.csv.gz"
        if not path.exists():
            raise SystemExit(f"Chưa có {path.relative_to(ROOT)}.\nHãy chạy qa.py trước (python src/qa.py).")
        cot = COT_CAN[bang]
        df = pd.read_csv(path, dtype=str, usecols=lambda c: c in cot)
        for c in cot:
            if c not in df.columns:
                df[c] = pd.NA
        ket_qua.append(df[cot].reset_index(drop=True))
    return tuple(ket_qua)

def dong_bi_co_loai(proc, snap, bang, df):
    """True với dòng bị gắn cờ 'loại bỏ' trong file cờ do qa.py tạo."""
    path = proc / "flagged" / snap / f"{bang}_flags.csv.gz"
    if not path.exists():
        raise SystemExit(f"Chưa có {path.relative_to(ROOT)}.\nHãy chạy qa.py trước (python src/qa.py).")
    co = pd.read_csv(path, dtype=str, keep_default_na=False)
    cot = [c for c in CO_LOAI_BO[bang] if c in co.columns]
    for c in cot:
        co[c] = co[c].str.lower().eq("true")
    co = co[KHOA[bang] + cot].drop_duplicates(KHOA[bang])
    ghep = df[KHOA[bang]].astype(str).merge(co, on=KHOA[bang], how="left")
    return pd.Series(ghep[cot].fillna(False).any(axis=1).values, index=df.index)

def lam_sach(proc, snap, listings, calendar):
    """Sau QA: liên kết dữ liệu phan_cong với file cờ, bỏ dòng bị cờ loại bỏ, dòng trùng khóa và lịch của listing đã bị loại."""
    l = listings[~(dong_bi_co_loai(proc, snap, "listings", listings) | listings.duplicated("id"))]
    c = calendar[~(dong_bi_co_loai(proc, snap, "calendar", calendar) | calendar.duplicated(["listing_id", "date"]))]
    c = c[c["listing_id"].isin(set(l["id"]))]
    return l.reset_index(drop=True), c.reset_index(drop=True)

def doc_llm(path, chi_giu_nhan_hop_le):
    """Đọc llm_cache.jsonl (mỗi dòng có review_id và overall_sentiment). Sau QA chỉ giữ nhãn hợp lệ."""
    if not path.exists():
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for dong_json in f:
            if dong_json.strip():
                it = json.loads(dong_json)
                rows.append({"review_id": str(it.get("review_id")),
                             "nhan_llm": str(it.get("overall_sentiment")).strip().lower()})
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("review_id", keep="last")
    return df[df["nhan_llm"].isin(SENTIMENTS)] if chi_giu_nhan_hop_le else df

def doc_nhan_tay(chi_giu_nhan_hop_le):
    path = ROOT / "data/samples/hand_labeled.csv"
    if not path.exists():
        return None
    hand = pd.read_csv(path, dtype=str, encoding="utf-8-sig")[["review_id", "label"]].dropna()
    hand["label"] = hand["label"].str.strip().str.lower()
    return hand[hand["label"].isin(SENTIMENTS)] if chi_giu_nhan_hop_le else hand

# 2. TÍNH 7 KPI: mỗi KPI trả về (giá trị, tên nhóm, tử số, mẫu số)
def pct(tu, mau):
    return round(tu / mau * 100, 2) if mau else None


def tinh_kpi_1_den_5(l, c):
    kq = {}
    # KPI1: đếm listing theo neighbourhood_cleansed -> một giá trị: số listing của khu vực lớn nhất (kèm tên)
    theo_khu_vuc = l.dropna(subset=[AREA_COL]).groupby(AREA_COL).size()
    if len(theo_khu_vuc):
        kq["KPI1"] = (int(theo_khu_vuc.max()), theo_khu_vuc.idxmax(), int(theo_khu_vuc.max()), int(theo_khu_vuc.sum()))
    else:
        kq["KPI1"] = (None, "", None, 0)

    # KPI2: số listing từng room_type / tổng listing -> một giá trị: tỷ lệ của loại phòng lớn nhất (kèm tên)
    dem = {rt: int((l["room_type"] == rt).sum()) for rt in ROOM_TYPES}
    top = max(dem, key=dem.get)
    kq["KPI2"] = (pct(dem[top], len(l)), top, dem[top], len(l))

    # KPI3: tỷ lệ listing chưa có review; KPI4: review trung bình trên listing
    so_review = pd.to_numeric(l["number_of_reviews"], errors="coerce")
    chua_co = int((so_review == 0).sum())
    kq["KPI3"] = (pct(chua_co, len(l)), "", chua_co, len(l))
    tong = float(so_review.sum())
    kq["KPI4"] = (round(tong / len(l), 2) if len(l) else None, "", tong, len(l))

    # KPI5: tỷ lệ ngày có available = 't'
    con_trong = int(c["available"].astype("string").str.strip().str.lower().eq("t").sum())
    kq["KPI5"] = (pct(con_trong, len(c)), "", con_trong, len(c))
    return kq


def diem_f1(nhan_tay, nhan_llm, cac_lop):
    """precision/recall/F1 từng lớp, macro F1, accuracy và ma trận nhầm lẫn (hàng = nhãn tay, cột = LLM)."""
    cm = pd.crosstab(pd.Series(list(nhan_tay), name="t"), pd.Series(list(nhan_llm), name="p"))
    cm = cm.reindex(index=cac_lop, columns=cac_lop, fill_value=0)
    tung_lop, f1_cac_lop = {}, []
    for k in cac_lop:
        tp = int(cm.loc[k, k])
        fp, fn = int(cm[k].sum()) - tp, int(cm.loc[k].sum()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        tung_lop[k] = (round(p, 4), round(r, 4), round(f1, 4), tp + fn)
        if tp + fn:
            f1_cac_lop.append(f1)                  # macro F1 chỉ tính lớp có xuất hiện trong nhãn tay
    tong = int(cm.values.sum())
    acc = sum(int(cm.loc[k, k]) for k in cac_lop) / tong if tong else None
    macro = sum(f1_cac_lop) / len(f1_cac_lop) if f1_cac_lop else None
    return tung_lop, macro, acc, cm

def tinh_kpi_6_7(llm, hand):
    kq = {}
    if llm is None or llm.empty:
        return kq
    # KPI6: tỷ lệ review tích cực theo LLM
    tich_cuc = int((llm["nhan_llm"] == "positive").sum())
    kq["KPI6"] = (pct(tich_cuc, len(llm)), "", tich_cuc, len(llm))
    # KPI7: macro F1 so với nhãn tay (chỉ review có cả hai nhãn)
    if hand is not None:
        gh = hand.merge(llm, on="review_id")
        if len(gh):
            macro = diem_f1(gh["label"], gh["nhan_llm"], sorted(set(gh["label"]) | set(gh["nhan_llm"])))[1]
            kq["KPI7"] = (None if macro is None else round(macro, 4), "", None, len(gh))
    return kq

# 3. DỮ LIỆU CHO viz.py (tính sẵn ở đây, viz.py chỉ vẽ)
def du_lieu_fig1(l, c, snap):
    """Hình 1 (bản đồ): số listing và tỷ lệ ngày còn trống theo từng khu vực, sau QA."""
    so_listing = l.dropna(subset=[AREA_COL]).groupby(AREA_COL).size().rename("so_listing")
    trong = c["available"].astype("string").str.strip().str.lower().eq("t")
    khu_vuc = l.dropna(subset=["id", AREA_COL]).drop_duplicates("id")[["id", AREA_COL]]
    lich = (c.assign(con_trong=trong).merge(khu_vuc, left_on="listing_id", right_on="id")
            .groupby(AREA_COL)["con_trong"].agg(["sum", "count"]))
    lich.columns = ["ngay_con_trong", "tong_ngay"]
    kq = pd.concat([so_listing, lich], axis=1).reset_index().rename(columns={AREA_COL: "khu_vuc"})
    kq["ty_le_ngay_con_trong_pct"] = [pct(a, b) if b else None for a, b in zip(kq["ngay_con_trong"], kq["tong_ngay"])]
    kq.insert(0, "snapshot", snap)
    return kq

def du_lieu_fig2(du_lieu_sau):
    """Hình 2 (theo thời gian): KPI 3, 4, 5 của từng snapshot, sau QA."""
    rows = []
    for snap, (l, c) in du_lieu_sau.items():
        kq = tinh_kpi_1_den_5(l, c)
        rows.append({"snapshot": snap, "ty_le_chua_review_pct": kq["KPI3"][0],
                     "review_tb_moi_listing": kq["KPI4"][0], "ty_le_ngay_con_trong_pct": kq["KPI5"][0]})
    return pd.DataFrame(rows)

def du_lieu_fig3(llm_sau, hand_sau):
    """Hình 3 (LLM): ma trận nhầm lẫn và F1 từng lớp của LLM so với nhãn tay, sau QA. None nếu chưa có kết quả."""
    if llm_sau is None or hand_sau is None:
        return None
    gh = hand_sau.merge(llm_sau, on="review_id")
    if gh.empty:
        return None
    tung_lop, macro, acc, cm = diem_f1(gh["label"], gh["nhan_llm"], SENTIMENTS)
    rows = [{"loai": "confusion", "nhan_that": t, "nhan_llm": p, "chi_so": "so_review", "gia_tri": int(cm.loc[t, p]),
             "so_mau": len(gh)} for t in SENTIMENTS for p in SENTIMENTS]
    for k, (p, r, f1, n) in tung_lop.items():
        rows += [{"loai": "lop", "nhan_that": k, "nhan_llm": "", "chi_so": ten, "gia_tri": v, "so_mau": n}
                 for ten, v in (("precision", p), ("recall", r), ("f1", f1))]
    rows += [{"loai": "tong", "nhan_that": "", "nhan_llm": "", "chi_so": "macro_f1", "gia_tri": macro, "so_mau": len(gh)},
             {"loai": "tong", "nhan_that": "", "nhan_llm": "", "chi_so": "accuracy", "gia_tri": acc, "so_mau": len(gh)}]
    return pd.DataFrame(rows)

# 4. IN BẢNG RA TERMINAL VÀ GHI CSV
def dinh_dang_so(v, so_le):
    if v is None or pd.isna(v):
        return "chưa có"
    return f"{v:,.0f}" if so_le == 0 else f"{v:,.{so_le}f}"

def co_so_tinh(kid, kq):
    """Cột 'Cơ sở tính': tử số / mẫu số (và tên nhóm với KPI1, KPI2) để biết con số lấy từ đâu."""
    _, chi_tiet, tu_so, mau_so = kq.get(kid, (None, "", None, None))
    if mau_so is None:
        return ""
    doi_tuong = "ngày" if kid == "KPI5" else ("review" if kid in ("KPI6", "KPI7") else "listing")
    if kid == "KPI1":
        return f"{chi_tiet} (lớn nhất) / {mau_so:,} {doi_tuong}"
    if kid == "KPI2":
        return f"{chi_tiet} (lớn nhất): {tu_so:,} / {mau_so:,} {doi_tuong}"
    if kid == "KPI7":
        return f"{mau_so:,} review có đủ 2 nhãn"
    return f"{tu_so:,.0f} / {mau_so:,} {doi_tuong}"

def in_bang(tieu_de, headers, rows, can_phai, cot_co_gian):
    """
    In bảng căn lề, chia cột. Khung ASCII để terminal nào cũng hiển thị đúng.
    Bảng tự co theo bề rộng terminal: các cột trong `cot_co_gian` được xuống dòng khi quá dài.
    """
    chuan = lambda x: unicodedata.normalize("NFC", str(x))
    headers = [chuan(h) for h in headers]
    rows = [[chuan(x) for x in r] for r in rows]
    tu_nhien = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    kha_dung = shutil.get_terminal_size((100, 24)).columns - (3 * len(headers) + 1) - 1
    rong = list(tu_nhien)
    if sum(rong) > kha_dung:
        co_dinh = sum(w for i, w in enumerate(tu_nhien) if i not in cot_co_gian)
        con_lai = max(kha_dung - co_dinh, 18 * len(cot_co_gian))
        tong_gian = sum(tu_nhien[i] for i in cot_co_gian)
        for i in cot_co_gian:
            rong[i] = max(14, int(con_lai * tu_nhien[i] / tong_gian))

    def dong_ngang():
        return "+" + "+".join("-" * (w + 2) for w in rong) + "+"

    def in_hang(cells):
        cac_dong = [textwrap.wrap(c, width=w) or [""] for c, w in zip(cells, rong)]
        for k in range(max(len(x) for x in cac_dong)):
            out = []
            for i, (nhieu_dong, w) in enumerate(zip(cac_dong, rong)):
                c = nhieu_dong[k] if k < len(nhieu_dong) else ""
                out.append(c.rjust(w) if i in can_phai else c.ljust(w))
            print("| " + " | ".join(out) + " |")

    print()
    for dong in tieu_de:
        print(dong)
    print(dong_ngang())
    in_hang(headers)
    print(dong_ngang())
    for r in rows:
        in_hang(r)
        print(dong_ngang())

def bang_kpi_csv(kq, snap):
    """File KPI: đúng 7 dòng, mỗi KPI một giá trị (KPI chưa tính được để trống)."""
    rows = []
    for kid, meta in KPI.items():
        gia_tri, nhom, tu_so, mau_so = kq.get(kid, (None, "", None, None))
        rows.append({"stt": meta["stt"], "kpi": meta["ten"], "cong_thuc": meta["cong_thuc"], "don_vi": meta["don_vi"],
                     "gia_tri": gia_tri, "nhom": nhom, "tu_so": tu_so, "mau_so": mau_so,
                     "pham_vi": snap if kid in ("KPI1", "KPI2", "KPI3", "KPI4", "KPI5") else "mẫu review đã chạy LLM"})
    return pd.DataFrame(rows)

def run(config_path=None, snapshot=None, skip_llm=False, llm_cache=None):
    try:
        sys.stdout.reconfigure(errors="replace")       # tránh lỗi mã hóa trên một số terminal Windows
    except (AttributeError, ValueError):
        pass
    cfg = doc_config(config_path)
    raw_dir, proc = thu_muc(cfg, "raw_dir", "data/raw"), thu_muc(cfg, "processed_dir", "data/processed")
    proc.mkdir(parents=True, exist_ok=True)
    snap = str(snapshot) if snapshot else sorted(cfg["snapshots"])[-1]
    if snap not in cfg["snapshots"]:
        raise SystemExit(f"Snapshot {snap} không có trong config. Các mốc hợp lệ: {', '.join(cfg['snapshots'])}")

    # ---- trước QA: chỉ cần data/raw/phan_cong của snapshot được chọn ----
    l_truoc, c_truoc = doc_truoc_qa(raw_dir, snap)
    # ---- sau QA: phan_cong liên kết với flagged; cần đủ các snapshot cho hình theo thời gian ----
    du_lieu_sau = {}
    for s in cfg["snapshots"]:
        l, c = (l_truoc, c_truoc) if s == snap else doc_truoc_qa(raw_dir, s)
        du_lieu_sau[s] = lam_sach(proc, s, l, c)
    l_sau, c_sau = du_lieu_sau[snap]
    truoc, sau = tinh_kpi_1_den_5(l_truoc, c_truoc), tinh_kpi_1_den_5(l_sau, c_sau)

    # ---- KPI 6, 7 ----
    llm_sau = hand_sau = None
    if skip_llm:
        print("(--skip-llm: bỏ KPI 6, 7)")
    else:
        cache = Path(llm_cache) if llm_cache else proc / "llm_cache.jsonl"
        cache = cache if cache.is_absolute() else ROOT / cache
        llm_truoc, llm_sau = doc_llm(cache, False), doc_llm(cache, True)
        hand_sau = doc_nhan_tay(True)
        if llm_truoc is None:
            print(f"(chưa có {cache.name}: KPI 6, 7 hiện 'chưa có'; chạy bước LLM trước hoặc dùng --llm-cache)")
        truoc.update(tinh_kpi_6_7(llm_truoc, doc_nhan_tay(False)))
        sau.update(tinh_kpi_6_7(llm_sau, hand_sau))

    # ---- bảng terminal: kết quả SAU QA ----
    rows = [[meta["stt"], meta["ten"], dinh_dang_so(sau.get(kid, (None,))[0], meta["so_le"]), meta["don_vi"],
             co_so_tinh(kid, sau)] for kid, meta in KPI.items()]
    in_bang([f"KẾT QUẢ 7 KPI SAU QA - {cfg.get('city', '')} - snapshot {snap}",
             f"Sau QA: {len(l_sau):,} listing, {len(c_sau):,} ngày lịch. KPI 6, 7: mẫu review đã chạy LLM"],
            ["#", "KPI", "Giá trị", "Đơn vị", "Cơ sở tính"], rows, can_phai={0, 2}, cot_co_gian=[1, 4])

    # ---- hai file KPI: trước QA và sau QA (mỗi file 7 dòng) ----
    p_truoc, p_sau = proc / "kpi_truoc_qa.csv", proc / "kpi_sau_qa.csv"
    bang_kpi_csv(truoc, snap).to_csv(p_truoc, index=False, encoding="utf-8-sig")
    bang_kpi_csv(sau, snap).to_csv(p_sau, index=False, encoding="utf-8-sig")

    # ---- dữ liệu cho viz.py ----
    viz_dir = proc / "viz"
    viz_dir.mkdir(exist_ok=True)
    du_lieu_fig1(l_sau, c_sau, snap).to_csv(viz_dir / "fig1_khu_vuc.csv", index=False, encoding="utf-8-sig")
    du_lieu_fig2(du_lieu_sau).to_csv(viz_dir / "fig2_theo_thoi_gian.csv", index=False, encoding="utf-8-sig")
    fig3 = du_lieu_fig3(llm_sau, hand_sau)
    if fig3 is not None:
        fig3.to_csv(viz_dir / "fig3_llm.csv", index=False, encoding="utf-8-sig")
    elif (viz_dir / "fig3_llm.csv").exists():
        (viz_dir / "fig3_llm.csv").unlink()               # tránh vẽ lại số liệu LLM cũ khi chưa có kết quả mới

    print(f"\nĐã ghi KPI trước QA : {p_truoc.relative_to(ROOT)}")
    print(f"Đã ghi KPI sau QA   : {p_sau.relative_to(ROOT)}")
    print(f"Dữ liệu cho viz.py  : {viz_dir.relative_to(ROOT)}")
    return bang_kpi_csv(truoc, snap), bang_kpi_csv(sau, snap)

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