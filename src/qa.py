"""QA: bộ quy tắc kiểm tra chất lượng và gắn cờ dòng vi phạm.

Nhiệm vụ duy nhất: nhận bảng đã được phân công (DataFrame) và trả về bảng cờ.
Chỉ gắn cờ, KHÔNG xóa/sửa dòng. Không đọc/ghi tệp, không tải dữ liệu,
không chọn dữ liệu theo phân công (việc của assign.py).
"""
from __future__ import annotations

import pandas as pd

TABLES = {"listings": ["id"], "calendar": ["listing_id", "date"], "reviews": ["id"]}
REQUIRED = {
    "listings": {"id", "analysis_area", "neighbourhood_cleansed", "room_type", "number_of_reviews", "number_of_reviews_ltm", "last_review", "price", "latitude", "longitude"},
    "calendar": {"listing_id", "date", "available"},
    "reviews": {"id", "date", "comments"},
}
ROOM_TYPES = ("Entire home/apt", "Private room", "Shared room", "Hotel room")
AREA_COLS = ["lon_min", "lat_min", "lon_max", "lat_max"]
AREA_SHEET_REQUIRED = {"Thành phố", "analysis_area", *AREA_COLS}

# rules: (name, condition, reason, todo)
RULES = [
    ("missing_neigh_cleansed", "neighbourhood_cleansed bị thiếu/None/null", "Không thể phân nhóm listing theo khu vực", "Gắn cờ"),
    ("missing_room_type", "room_type bị thiếu/None/null", "Không thể xác định loại phòng để tính tỷ lệ", "Gắn cờ"),
    ("missing_number_of_reviews", "number_of_reviews hoặc number_of_reviews_ltm bị thiếu/None/null", "Không thể xác định listing không có review 12 tháng qua hoặc tính số review trung bình", "Gắn cờ"),
    ("missing_price", "price bị thiếu/None/null (chỉ xét ở mốc cuối)", "Không thể sử dụng giá listing để tính giá thuê trung bình ở mốc cuối", "Gắn cờ"),
    ("missing_review_date", "date bị thiếu/None/null trong reviews", "Không thể xác định thời điểm review được tạo", "Gắn cờ"),
    ("missing_comments", "comments bị thiếu/None/null hoặc chỉ chứa khoảng trắng", "Không có nội dung để phân tích cảm xúc bằng LLM", "Gắn cờ và loại khỏi tập đầu vào LLM nếu không có nội dung"),
    ("invalid_reviews_num", "number_of_reviews hoặc number_of_reviews_ltm không phải số nguyên hợp lệ", "Không thể tính chính xác số review và các KPI liên quan", "Gắn cờ"),
    ("invalid_room_type", "room_type không thuộc danh mục loại phòng được chấp nhận", "Có thể làm sai tỷ lệ các loại phòng", "Gắn cờ"),
    ("invalid_price", "price không thể chuyển thành giá trị số hợp lệ hoặc bị âm (chỉ xét ở mốc cuối)", "Không thể sử dụng giá trong các phép tính", "Gắn cờ"),
    ("invalid_calendar_date", "date không chuyển được sang kiểu ngày hợp lệ", "Không thể phân tích lịch theo thời gian", "Gắn cờ"),
    ("invalid_calendar_available", "available không thuộc tập giá trị hợp lệ của nguồn dữ liệu, thường là t/f", "Không thể tính chính xác tỷ lệ ngày còn trống", "Gắn cờ"),
    ("invalid_review_date", "date của review không chuyển được sang kiểu ngày hợp lệ", "Không thể phân tích review theo thời gian", "Gắn cờ"),
    ("invalid_area", "tọa độ của id không khớp tọa độ vùng", "không thể vẽ biểu đồ vùng chính xác", "Gắn cờ"),
    ("outlier_price", "price nằm ngoài ngưỡng IQR (chỉ xét ở mốc cuối)", "Giá bất thường có thể ảnh hưởng đến phân tích giá", "Gắn cờ"),
    ("dup_listing_id", "id xuất hiện nhiều hơn một lần trong cùng snapshot listings", "Có thể làm tăng sai số lượng listing và ảnh hưởng các KPI", "Gắn cờ"),
    ("dup_cal_key", "Cặp (listing_id, date) xuất hiện nhiều hơn một lần trong cùng snapshot calendar", "Một listing có thể bị đếm nhiều lần trong cùng một ngày", "Gắn cờ"),
    ("calendar_days_per_listing_mismatch", "Một listing có số ngày hoặc số ngày duy nhất trong calendar khác 365", "Dữ liệu lịch không phủ đủ 365 ngày có thể làm sai tỷ lệ ngày đặt phòng", "Gắn cờ"),
    ("dup_review_id_in_snapshot", "ID review xuất hiện nhiều hơn một lần trong cùng snapshot reviews", "Có thể khiến một review bị đếm nhiều lần", "Gắn cờ"),
    ("dup_review_id_across_snapshots", "ID review xuất hiện ở nhiều snapshot khác nhau", "Review cũ thường được giữ lại ở snapshot sau; cần kiểm tra nhưng không tự loại", "Gắn cờ"),
    ("reviews_inconsistent", "number_of_reviews = 0 nhưng có review_scores_*/last_review, hoặc ngược lại", "Mâu thuẫn", "Gắn cờ"),
]

RULE_META = {
    "missing_neigh_cleansed": ("listings", ["KPI1", "KPI2"]),
    "missing_room_type": ("listings", ["KPI2"]),
    "missing_number_of_reviews": ("listings", ["KPI3", "KPI4"]),
    "missing_price": ("listings", ["KPI6"]),
    "invalid_reviews_num": ("listings", ["KPI3", "KPI4"]),
    "invalid_room_type": ("listings", ["KPI2"]),
    "invalid_price": ("listings", ["KPI6"]),
    "invalid_area": ("listings", []),
    "outlier_price": ("listings", ["KPI6"]),
    "dup_listing_id": ("listings", ["KPI1", "KPI2", "KPI3", "KPI4", "KPI6"]),
    "reviews_inconsistent": ("listings", ["KPI3", "KPI4"]),
    "invalid_calendar_date": ("calendar", ["KPI5"]),
    "invalid_calendar_available": ("calendar", ["KPI5"]),
    "dup_cal_key": ("calendar", ["KPI5"]),
    "calendar_days_per_listing_mismatch": ("calendar", ["KPI5"]),
    "missing_review_date": ("reviews", []),
    "missing_comments": ("reviews", ["KPI7", "KPI8"]),
    "invalid_review_date": ("reviews", []),
    "dup_review_id_in_snapshot": ("reviews", ["KPI7", "KPI8"]),
    "dup_review_id_across_snapshots": ("reviews", ["KPI7", "KPI8"]),
}


def _missing(s):
    return s.isna() | s.astype("string").str.strip().str.lower().isin(["", "none", "null", "nan"])


def _num(s, pat=r"[$,]", rep=""):
    return pd.to_numeric(s.astype("string").str.replace(pat, rep, regex=True).str.strip(), errors="coerce")


def _invalid_price(s):
    price = _num(s)
    return ~_missing(s) & (price.isna() | price.lt(0))


def _flag(df, keys, rules):
    f = df[keys].astype(str)
    for name, v in rules.items():
        f[name] = pd.Series(v, index=f.index).fillna(False).astype(bool)
    return f


def parse_area_bounds(area: pd.DataFrame, city: str) -> pd.DataFrame:
    """Bảng tọa độ min-max của từng analysis_area thuộc thành phố."""
    a = area.loc[area["Thành phố"].str.strip().eq(str(city).strip())].copy()
    assert not a.empty, f"Không tìm thấy khu vực của thành phố {city}"
    a["analysis_area"] = a["analysis_area"].str.strip()
    assert not a["analysis_area"].eq("").any(), "Sheet area có analysis_area bị thiếu"
    assert not a["analysis_area"].duplicated().any(), "Sheet area có analysis_area bị trùng"
    for c in AREA_COLS:
        a[c] = _num(a[c], ",", ".")
    assert not a[AREA_COLS].isna().any().any(), "Sheet area có tọa độ min-max không hợp lệ"
    assert not (a["lon_min"].gt(a["lon_max"]) | a["lat_min"].gt(a["lat_max"])).any(), "Tọa độ min-max của area không hợp lệ"
    return a.set_index("analysis_area")


def _invalid_area(L, bounds):
    area = L["analysis_area"].str.strip()
    lat, lon = _num(L["latitude"], ",", "."), _num(L["longitude"], ",", ".")
    b = {c: area.map(bounds[c]) for c in AREA_COLS}
    bad = lat.isna() | lon.isna() | pd.concat(b.values(), axis=1).isna().any(axis=1)
    return bad | lat.lt(b["lat_min"]) | lat.gt(b["lat_max"]) | lon.lt(b["lon_min"]) | lon.gt(b["lon_max"])


def self_check(qa: dict, snapshot: str, listings: pd.DataFrame, bounds: pd.DataFrame) -> None:
    """Đối chiếu dữ liệu đã lọc với số liệu kỳ vọng khai trong cấu hình qa."""
    s, L = snapshot, listings
    rt = L["room_type"].str.strip().value_counts().to_dict()
    assert rt == qa["room_types"], f"{s}: room_type {rt} khác cấu hình {qa['room_types']}"
    if s in qa.get("number_of_reviews", {}):
        zero = int(_num(L["number_of_reviews"]).eq(0).sum())
        assert zero == qa["number_of_reviews"][s], f"{s}: có {zero} listing number_of_reviews = 0, cần {qa['number_of_reviews'][s]}"
    if s in qa.get("missing_price", {}):
        n = int(_missing(L["price"]).sum())
        assert n == qa["missing_price"][s], f"{s}: có {n} listing thiếu price, cần {qa['missing_price'][s]}"
    c = L["analysis_area"].str.strip().value_counts()
    assert set(c.index) == set(bounds.index) and c.between(qa["area_min"], qa["area_max"]).all(), \
        f"{s}: vùng không khớp sheet area hoặc số listing/vùng ngoài [{qa['area_min']}, {qa['area_max']}]: {c.to_dict()}"


def run_qa(data: dict, bounds: pd.DataFrame, qa: dict, cal_days: int, log=lambda msg: None) -> dict:
    """Gắn cờ toàn bộ dữ liệu.

    data: {snapshot: {'listings', 'calendar', 'reviews': DataFrame}} (cột kiểu chuỗi).
    Trả về {(snapshot, table): DataFrame cờ} gồm cột khóa + một cột bool mỗi quy tắc.
    """
    snaps = list(data)
    ids = pd.concat([t["reviews"]["id"].drop_duplicates() for t in data.values()])
    review_cnt = ids[ids.str.strip().ne("")].value_counts()

    frames = {}
    for s, t in data.items():
        log(f"Đang gắn cờ mốc {s}!")
        L, C, R = t["listings"], t["calendar"], t["reviews"]
        self_check(qa, s, L, bounds)
        use_price = s == max(snaps)  # chỉ dùng price ở mốc cuối
        price = _num(L["price"])
        room = L["room_type"].astype("string").str.strip()
        nrev = L["number_of_reviews"].astype("string").str.strip()
        nrev_ltm = L["number_of_reviews_ltm"].astype("string").str.strip()
        has_info = pd.concat([~_missing(L[c]) for c in ["last_review", *L.filter(regex="^review_scores_").columns]], axis=1).any(axis=1)
        # không có giá hợp lệ: (0, 0) vẫn cho outlier = False vì mọi phép so sánh với NA đều bị fillna(False)
        valid_price = price[price.notna() & price.ge(0)]
        q1, q3 = valid_price.quantile([0.25, 0.75]) if not valid_price.empty else (0, 0)
        iqr = q3 - q1
        frames[s, "listings"] = _flag(L, ["id"], {
            "missing_neigh_cleansed": _missing(L["neighbourhood_cleansed"]),
            "missing_room_type": _missing(L["room_type"]),
            "missing_number_of_reviews": _missing(L["number_of_reviews"]) | _missing(L["number_of_reviews_ltm"]),
            "missing_price": _missing(L["price"]) & use_price,
            "invalid_reviews_num": (~_missing(L["number_of_reviews"]) & ~nrev.str.fullmatch(r"\d+", na=False))
                                   | (~_missing(L["number_of_reviews_ltm"]) & ~nrev_ltm.str.fullmatch(r"\d+", na=False)),
            "invalid_room_type": ~_missing(L["room_type"]) & ~room.isin(ROOM_TYPES),
            "invalid_price": _invalid_price(L["price"]) & use_price,
            "invalid_area": _invalid_area(L, bounds),
            "outlier_price": ((price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)) & use_price,
            "dup_listing_id": L["id"].duplicated(keep=False),
            "reviews_inconsistent": _num(L["number_of_reviews"]).gt(0).ne(has_info),
        })
        av = C["available"].astype("string").str.strip().str.lower()
        g = C.groupby("listing_id")
        frames[s, "calendar"] = _flag(C, ["listing_id", "date"], {
            "invalid_calendar_date": ~_missing(C["date"]) & pd.to_datetime(C["date"], errors="coerce").isna(),
            "invalid_calendar_available": ~_missing(C["available"]) & ~av.isin({"t", "f"}),
            "dup_cal_key": C.duplicated(subset=["listing_id", "date"], keep=False),
            "calendar_days_per_listing_mismatch": g["date"].transform("size").ne(cal_days) | g["date"].transform("nunique").ne(cal_days),
        })
        frames[s, "reviews"] = _flag(R, ["id"], {
            "missing_review_date": _missing(R["date"]),
            "missing_comments": _missing(R["comments"]),
            "invalid_review_date": ~_missing(R["date"]) & pd.to_datetime(R["date"], errors="coerce").isna(),
            "dup_review_id_in_snapshot": R["id"].duplicated(keep=False),
            "dup_review_id_across_snapshots": R["id"].map(review_cnt).fillna(0).gt(1),
        })
    return frames


def _fmt_kpis(kpi_dict: dict, kids: list[str], currency: str) -> str:
    """Định dạng ngắn gọn các KPI chịu tác động của một quy tắc QA."""
    if not kids:
        return ""
    parts = []
    for kid in kids:
        val, _, _, den = kpi_dict.get(kid, (None, "", None, None))
        if val is None or pd.isna(val):
            parts.append(f"{kid}=chưa có")
        elif kid == "KPI1":
            parts.append(f"{kid}={val:,.0f}")
        elif kid in ("KPI2", "KPI3", "KPI5", "KPI7"):
            parts.append(f"{kid}={val:.2f}%")
        elif kid == "KPI4":
            parts.append(f"{kid}={val:.2f}")
        elif kid == "KPI6":
            parts.append(f"{kid}={val:,.2f} {currency}/đêm (n={den:,})")
        elif kid == "KPI8":
            parts.append(f"{kid}={val:.4f}")
    return "; ".join(parts)


def build_qa_report(flags: dict, kpi_before: dict, kpi_after: dict, currency: str = "CLP") -> pd.DataFrame:
    """Tổng hợp tác động QA theo từng mốc chụp và quy tắc (số dòng gắn cờ; KPI trước và sau)."""
    snapshots = list(dict.fromkeys(s for s, _ in flags))
    rows = []
    for s in snapshots:
        before_s = kpi_before.get(s, {})
        after_s = kpi_after.get(s, {})
        for name, condition, reason, action in RULES:
            table, affected = RULE_META[name]
            df = flags[s, table]
            total_rows = len(df)
            flagged_rows = int(df[name].sum()) if name in df.columns else 0
            flagged_pct = round(flagged_rows / total_rows * 100, 2) if total_rows else 0.0
            rows.append({
                "snapshot": s,
                "table": table,
                "rule": name,
                "condition": condition,
                "reason": reason,
                "action": action,
                "total_rows": total_rows,
                "flagged_rows": flagged_rows,
                "flagged_pct": flagged_pct,
                "affected_kpis": ", ".join(affected),
                "kpi_before_qa": _fmt_kpis(before_s, affected, currency),
                "kpi_after_qa": _fmt_kpis(after_s, affected, currency),
            })
    return pd.DataFrame(rows)
