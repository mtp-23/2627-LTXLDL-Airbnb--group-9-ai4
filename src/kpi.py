"""Tính 8 KPI trước và sau QA phục vụ Chính quyền thành phố.

Nhiệm vụ duy nhất: từ bảng listings/calendar (và kết quả LLM, F1 do evaluate.py
tính sẵn) tính KPI, bảng KPI theo khu vực/thời gian, và định dạng bảng hiển thị.
Không đọc/ghi tệp, không đọc cấu hình, không tự tính F1, không vẽ hình.
"""
from __future__ import annotations

import shutil
import textwrap
import unicodedata

import pandas as pd

# Cờ QA làm dòng bị LOẠI khỏi dữ liệu sạch (tên khớp với RULES trong qa.py)
EXCLUDE_FLAGS = {
    "listings": ["missing_neigh_cleansed", "missing_room_type", "missing_number_of_reviews",
                 "invalid_reviews_num", "invalid_room_type"],
    "calendar": ["invalid_calendar_date", "invalid_calendar_available"],
}
PRICE_EXCLUDE_FLAGS = ["missing_price", "invalid_price", "outlier_price"]
KEY_COLUMNS = {"listings": ["id"], "calendar": ["listing_id", "date"]}
SNAPSHOT_KPIS = ("KPI1", "KPI2", "KPI3", "KPI4", "KPI5", "KPI6")

# 8 KPI phục vụ Chính quyền thành phố (đầy đủ tên, mục tiêu, công thức, đơn vị)
KPI = {
    "KPI1": {"number": 1, "name": "Số lượng listing theo khu vực",
             "target": "Xác định địa bàn tập trung nguồn cung lưu trú để phân bổ nguồn lực quản lý đô thị và hạ tầng",
             "formula": "Đếm listing theo neighbourhood_cleansed",
             "unit": "Phòng", "decimals": 0},
    "KPI2": {"number": 2, "name": "Tỷ lệ nhà nguyên căn tại khu vực trọng điểm",
             "target": "Đo mức độ thương mại hóa nhà ở dân cư tại khu vực tập trung nhiều listing nhất",
             "formula": "Số listing Entire home/apt / tổng listing tại khu vực lớn nhất × 100%",
             "unit": "%", "decimals": 2},
    "KPI3": {"number": 3, "name": "Tỷ lệ listing không có review trong 12 tháng qua",
             "target": "Nhận diện tỷ lệ listing ngủ đông không phát sinh đánh giá trong 1 năm gần nhất",
             "formula": "Listing có number_of_reviews_ltm = 0 / tổng listing × 100%",
             "unit": "%", "decimals": 2},
    "KPI4": {"number": 4, "name": "Số review trung bình trên listing",
             "target": "Theo dõi quy mô thu hút du khách tích lũy và mức độ tương tác qua các quý",
             "formula": "Tổng number_of_reviews / tổng listing",
             "unit": "Review/listing", "decimals": 2},
    "KPI5": {"number": 5, "name": "Tỷ lệ ngày có thể đặt phòng",
             "target": "Đo độ mở của nguồn cung phòng trong 365 ngày tới và biến động mùa vụ",
             "formula": "Số ngày available = 't' / tổng ngày hợp lệ × 100%",
             "unit": "%", "decimals": 2},
    "KPI6": {"number": 6, "name": "Giá thuê phòng trung bình một đêm (mốc cuối)",
             "target": "Nắm mặt bằng giá lưu trú thực tế để xây dựng khung thuế du lịch và đánh giá khả năng chi trả",
             "formula": "Tổng price hợp lệ / số listing có giá hợp lệ ở mốc cuối",
             "unit": "{currency}/đêm", "decimals": 2},
    "KPI7": {"number": 7, "name": "Tỷ lệ review tích cực theo LLM",
             "target": "Đánh giá mức độ hài lòng tổng thể của du khách đối với chất lượng lưu trú",
             "formula": "Review được LLM phân loại tích cực / tổng review có nhãn hợp lệ × 100%",
             "unit": "%", "decimals": 2},
    "KPI8": {"number": 8, "name": "F1-score của LLM",
             "target": "Kiểm định độ tin cậy của mô hình LLM đối chiếu với nhãn thủ công",
             "formula": "So sánh nhãn LLM với nhãn thủ công",
             "unit": "Điểm F1 (0-1)", "decimals": 4},
}


# 1. LÀM SẠCH THEO CỜ QA
def _flagged_for_exclusion(df: pd.DataFrame, flags: pd.DataFrame, table: str, flag_cols=None) -> pd.Series:
    """True với dòng bị gắn cờ 'loại bỏ' (ghép theo cột khóa)."""
    keys = KEY_COLUMNS[table]
    target_cols = flag_cols if flag_cols is not None else EXCLUDE_FLAGS[table]
    columns = [c for c in target_cols if c in flags.columns]
    if not columns:
        return pd.Series(False, index=df.index)
    flags = flags[keys + columns].drop_duplicates(keys)
    merged = df[keys].astype(str).merge(flags, on=keys, how="left")
    return pd.Series(merged[columns].fillna(False).astype(bool).any(axis=1).values, index=df.index)


def clean_after_qa(listings, calendar, listing_flags, calendar_flags):
    """Bỏ dòng bị cờ loại bỏ, dòng trùng khóa và lịch của listing đã bị loại."""
    l = listings[~(_flagged_for_exclusion(listings, listing_flags, "listings") | listings.duplicated("id"))]
    c = calendar[~(_flagged_for_exclusion(calendar, calendar_flags, "calendar") | calendar.duplicated(["listing_id", "date"]))]
    c = c[c["listing_id"].isin(set(l["id"]))]
    return l.reset_index(drop=True), c.reset_index(drop=True)


# 2. TÍNH 8 KPI: mỗi KPI là (giá trị, tên nhóm, tử số, mẫu số)
def pct(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else None


def _with_area(l: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Listing có khu vực không trống."""
    return l[l[area_col].notna() & l[area_col].str.strip().ne("")]


def _numeric_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"[$,]", "", regex=True).str.strip(), errors="coerce")


def compute_kpis(l, c, area_col, room_types, llm=None, llm_f1=None, listing_flags=None):
    """KPI 1-6 từ listings/calendar; KPI7 từ `llm` (cột sentiment); KPI8 từ `llm_f1` = (macro-F1, số review chấm).

    Khi truyền `listing_flags` (sau QA), KPI6 loại bỏ các dòng bị cờ missing_price / invalid_price / outlier_price.
    """
    result = {}
    # KPI1: số listing của khu vực lớn nhất (kèm tên)
    valid_area = _with_area(l, area_col)
    by_area = valid_area.groupby(area_col).size()
    if len(by_area):
        top_area = by_area.idxmax()
        top_count = int(by_area.max())
        result["KPI1"] = (top_count, top_area, top_count, int(by_area.sum()))
        # KPI2: tỷ lệ nhà nguyên căn (Entire home/apt) tại khu vực trọng điểm (đông listing nhất)
        top_listings = valid_area[valid_area[area_col] == top_area]
        entire_in_top = int(top_listings["room_type"].astype("string").str.strip().eq("Entire home/apt").sum())
        result["KPI2"] = (pct(entire_in_top, top_count), f"{top_area} (Entire home/apt)", entire_in_top, top_count)
    else:
        result["KPI1"] = (None, "", None, 0)
        result["KPI2"] = (None, "", None, 0)

    # KPI3: tỷ lệ listing không có review trong 12 tháng qua (number_of_reviews_ltm = 0)
    ltm_counts = pd.to_numeric(l["number_of_reviews_ltm"], errors="coerce")
    inactive_12m = int((ltm_counts == 0).sum())
    result["KPI3"] = (pct(inactive_12m, len(l)), "", inactive_12m, len(l))

    # KPI4: số review trung bình trên listing
    review_counts = pd.to_numeric(l["number_of_reviews"], errors="coerce")
    total_reviews = float(review_counts.sum())
    result["KPI4"] = (round(total_reviews / len(l), 2) if len(l) else None, "", total_reviews, len(l))

    # KPI5: tỷ lệ ngày có available = 't'
    available_days = int(c["available"].astype("string").str.strip().str.lower().eq("t").sum())
    result["KPI5"] = (pct(available_days, len(c)), "", available_days, len(c))

    # KPI6: giá thuê phòng trung bình một đêm (mốc cuối); sau QA loại thêm cờ giá (missing/invalid/outlier)
    price_df = l[~_flagged_for_exclusion(l, listing_flags, "listings", PRICE_EXCLUDE_FLAGS)] if listing_flags is not None else l
    prices = _numeric_price(price_df["price"])
    valid_prices = prices[prices.notna() & prices.ge(0)]
    if len(valid_prices):
        total_price = float(valid_prices.sum())
        result["KPI6"] = (round(total_price / len(valid_prices), 2), "", total_price, len(valid_prices))
    else:
        result["KPI6"] = (None, "", None, 0)

    # KPI7: tỷ lệ review tích cực theo LLM
    if llm is not None and len(llm):
        positive = int((llm["sentiment"] == "positive").sum())
        result["KPI7"] = (pct(positive, len(llm)), "", positive, len(llm))
    # KPI8: macro-F1 do evaluate.py tính
    if llm_f1 is not None:
        f1, n = llm_f1
        result["KPI8"] = (round(f1, 4), "", None, n)
    return result


# 3. KPI THEO KHU VỰC / THỜI GIAN (đầu vào của viz.py)
def kpi_by_area(l, c, snapshot, area_col):
    """Số listing, tỷ lệ nhà nguyên căn và tỷ lệ ngày còn trống theo từng khu vực (sau QA)."""
    valid = _with_area(l, area_col)
    listing_counts = valid.groupby(area_col).size().rename("listing_count")
    entire_counts = (valid["room_type"].astype("string").str.strip().eq("Entire home/apt")
                     .groupby(valid[area_col]).sum().astype(int).rename("entire_home_count"))
    is_available = c["available"].astype("string").str.strip().str.lower().eq("t")
    area_lookup = valid.drop_duplicates("id")[["id", area_col]]
    calendar_stats = (c.assign(is_available_day=is_available).merge(area_lookup, left_on="listing_id", right_on="id")
                      .groupby(area_col)["is_available_day"].agg(["sum", "count"]))
    calendar_stats.columns = ["available_days", "total_days"]
    result = pd.concat([listing_counts, entire_counts, calendar_stats], axis=1).reset_index().rename(columns={area_col: "area"})
    result["entire_home_rate_pct"] = [pct(e, n) if n else None for e, n in zip(result["entire_home_count"], result["listing_count"])]
    result["available_day_rate_pct"] = [pct(a, b) if b else None for a, b in zip(result["available_days"], result["total_days"])]
    result.insert(0, "snapshot", snapshot)
    return result


def kpi_over_time(data_after_qa, area_col, room_types):
    """KPI 3, 4, 5 của từng snapshot (sau QA). data_after_qa: {snapshot: (listings, calendar)}."""
    rows = []
    for snap, (l, c) in data_after_qa.items():
        r = compute_kpis(l, c, area_col, room_types)
        rows.append({"snapshot": snap, "no_review_rate_pct": r["KPI3"][0],
                     "avg_reviews_per_listing": r["KPI4"][0], "available_day_rate_pct": r["KPI5"][0]})
    return pd.DataFrame(rows)


# 4. BẢNG KPI VÀ ĐỊNH DẠNG HIỂN THỊ
def _unit(meta, currency: str) -> str:
    return meta["unit"].format(currency=currency)


def kpi_table(result, snapshot, currency: str = "CLP"):
    """Bảng KPI đúng 8 dòng, mỗi KPI một giá trị (kèm mục tiêu, công thức, đơn vị tiền tệ)."""
    rows = []
    for kid, meta in KPI.items():
        value, group, numerator, denominator = result.get(kid, (None, "", None, None))
        rows.append({"number": meta["number"], "kpi": meta["name"], "target": meta["target"],
                     "formula": meta["formula"], "unit": _unit(meta, currency),
                     "value": value, "group": group, "numerator": numerator, "denominator": denominator,
                     "scope": snapshot if kid in SNAPSHOT_KPIS else "mẫu review đã chạy LLM"})
    return pd.DataFrame(rows)


def _format_number(v, decimals):
    if v is None or pd.isna(v):
        return "chưa có"
    return f"{v:,.0f}" if decimals == 0 else f"{v:,.{decimals}f}"


def _calculation_basis(kid, result):
    """Cột 'Cơ sở tính': tử số / mẫu số (và tên nhóm với KPI1, KPI2)."""
    _, detail, numerator, denominator = result.get(kid, (None, "", None, None))
    if not denominator:
        return ""
    unit_label = "ngày" if kid == "KPI5" else ("review" if kid in ("KPI7", "KPI8") else "listing")
    if kid == "KPI1":
        return f"{detail} (lớn nhất) trong {denominator:,} {unit_label}"
    if kid == "KPI2":
        return f"{detail}: {numerator:,} / {denominator:,} {unit_label}"
    if kid == "KPI8":
        return f"{denominator:,} review có đủ 2 nhãn"
    return f"{numerator:,.0f} / {denominator:,} {unit_label}"


def render_table(result, title_lines, currency: str = "CLP"):
    """Bảng ASCII căn lề (chuỗi) cho 8 KPI; tự co theo bề rộng terminal."""
    headers = ["#", "KPI", "Giá trị", "Đơn vị", "Cơ sở tính"]
    rows = [[meta["number"], meta["name"], _format_number(result.get(kid, (None,))[0], meta["decimals"]),
             _unit(meta, currency), _calculation_basis(kid, result)] for kid, meta in KPI.items()]
    right_aligned, stretch_columns = {0, 2}, [1, 4]
    norm = lambda x: unicodedata.normalize("NFC", str(x))
    headers = [norm(h) for h in headers]
    rows = [[norm(x) for x in r] for r in rows]
    natural = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    available = shutil.get_terminal_size((100, 24)).columns - (3 * len(headers) + 1) - 1
    widths = list(natural)
    if sum(widths) > available:
        fixed = sum(w for i, w in enumerate(natural) if i not in stretch_columns)
        remaining = max(available - fixed, 18 * len(stretch_columns))
        stretch_total = sum(natural[i] for i in stretch_columns)
        for i in stretch_columns:
            widths[i] = max(14, int(remaining * natural[i] / stretch_total))

    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = ["", *title_lines, line]

    def add_row(cells):
        wrapped = [textwrap.wrap(c, width=w) or [""] for c, w in zip(cells, widths)]
        for k in range(max(len(x) for x in wrapped)):
            parts = []
            for i, (cell_lines, w) in enumerate(zip(wrapped, widths)):
                c = cell_lines[k] if k < len(cell_lines) else ""
                parts.append(c.rjust(w) if i in right_aligned else c.ljust(w))
            out.append("| " + " | ".join(parts) + " |")

    add_row(headers)
    out.append(line)
    for r in rows:
        add_row(r)
        out.append(line)
    return "\n".join(out)
