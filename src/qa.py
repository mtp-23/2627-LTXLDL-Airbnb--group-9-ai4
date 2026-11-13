import gzip
import os
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "santiago.yaml"
SHEETS = ("listing", "review", "area")
TABLES = {"listings": ["id"], "calendar": ["listing_id", "date"], "reviews": ["id"]}
REQUIRED = {
    "listings": {"id", "analysis_area", "neighbourhood_cleansed", "room_type", "number_of_reviews", "last_review", "price", "latitude", "longitude"},
    "calendar": {"listing_id", "date", "available"},
    "reviews": {"id", "date", "comments"},
}
ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}
AREA_COLS = ["lon_min", "lat_min", "lon_max", "lat_max"]
CAL_DAYS = 365  # calendar của Inside Airbnb luôn phủ 365 ngày/listing

# rules: (name, condition, reason, todo)
RULES = [
    ("missing_neigh_cleansed", "neighbourhood_cleansed bị thiếu/None/null", "Không thể phân nhóm listing theo khu vực", "Gắn cờ"),
    ("missing_room_type", "room_type bị thiếu/None/null", "Không thể xác định loại phòng để tính tỷ lệ", "Gắn cờ"),
    ("missing_number_of_reviews", "number_of_reviews bị thiếu/None/null", "Không thể xác định listing có review hay không hoặc tính số review trung bình", "Gắn cờ"),
    ("missing_price", "price bị thiếu/None/null (chỉ xét ở mốc cuối)", "Không thể sử dụng giá listing trong các phân tích cần đến giá", "Gắn cờ"),
    ("missing_review_date", "date bị thiếu/None/null trong reviews", "Không thể xác định thời điểm review được tạo", "Gắn cờ"),
    ("missing_comments", "comments bị thiếu/None/null hoặc chỉ chứa khoảng trắng", "Không có nội dung để phân tích cảm xúc bằng LLM", "Gắn cờ và loại khỏi tập đầu vào LLM nếu không có nội dung"),
    ("invalid_reviews_num", "number_of_reviews không phải số nguyên hợp lệ", "Không thể tính chính xác số review và các KPI liên quan", "Gắn cờ"),
    ("invalid_room_type", "room_type không thuộc danh mục loại phòng được chấp nhận", "Có thể làm sai tỷ lệ các loại phòng", "Gắn cờ"),
    ("invalid_price", "price không thể chuyển thành giá trị số hợp lệ hoặc bị âm (chỉ xét ở mốc cuối)", "Không thể sử dụng giá trong các phép tính", "Gắn cờ"),
    ("invalid_calendar_date", "date không chuyển được sang kiểu ngày hợp lệ", "Không thể phân tích lịch theo thời gian", "Gắn cờ"),
    ("invalid_calendar_available", "available không thuộc tập giá trị hợp lệ của nguồn dữ liệu, thường là t/f", "Không thể tính chính xác tỷ lệ ngày còn trống", "Gắn cờ"),
    ("invalid_review_date", "date của review không chuyển được sang kiểu ngày hợp lệ", "Không thể phân tích review theo thời gian", "Gắn cờ"),
    ("invalid_area", "tọa độ của id không khớp tọa độ vùng", "không thể vẽ biểu đồ vùng chính xác", "Gắn cờ"),
    ("outlier_price", "price nằm ngoài ngưỡng IQR (chỉ xét ở mốc cuối)", "Giá bất thường có thể ảnh hưởng đến phân tích giá", "Gắn cờ"),
    ("dup_listing_id", "id xuất hiện nhiều hơn một lần trong cùng snapshot listings", "Có thể làm tăng sai số lượng listing và ảnh hưởng các KPI", "Gắn cờ"),
    ("dup_cal_key", "Cặp (listing_id, date) xuất hiện nhiều hơn một lần trong cùng snapshot calendar", "Một listing có thể bị đếm nhiều lần trong cùng một ngày", "Gắn cờ"),
    ("dup_review_id_in_snapshot", "ID review xuất hiện nhiều hơn một lần trong cùng snapshot reviews", "Có thể khiến một review bị đếm nhiều lần", "Gắn cờ"),
    ("dup_review_id_across_snapshots", "ID review xuất hiện ở nhiều snapshot khác nhau", "Review cũ thường được giữ lại ở snapshot sau; cần kiểm tra nhưng không tự loại", "Gắn cờ"),
    ("reviews_inconsistent", "number_of_reviews = 0 nhưng có review_scores_*/last_review, hoặc ngược lại", "Mâu thuẫn", "Gắn cờ"),
]

def _atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def _write_csv(data, path):
    _atomic(path, lambda t: data.to_csv(t, index=False, encoding="utf-8", compression="gzip"))

def _read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def _snapshots(config):
    s = config.get("snapshots")
    assert isinstance(s, list) and s and len(set(s)) == len(s), "'snapshots' phải là danh sách không rỗng, không trùng"
    assert all(isinstance(x, str) and x.strip() for x in s), "Tên snapshot không hợp lệ"
    return s

def _read_config():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(config, dict), "File YAML không chứa cấu hình hợp lệ"
    snaps, qa = _snapshots(config), config.get("qa")
    assert isinstance(qa, dict), "Thiếu mục 'qa' trong file cấu hình"
    assert isinstance(qa.get("sheet_id"), str) and qa["sheet_id"].strip(), "Thiếu 'qa.sheet_id' trong file cấu hình"
    for k in ("gid", "key_col", "room_types", "number_of_reviews", "missing_price"):
        assert isinstance(qa.get(k), dict), f"Thiếu 'qa.{k}' trong file cấu hình"
    assert all(isinstance(qa.get(k), int) for k in ("area_min", "area_max")), "Thiếu 'qa.area_min'/'qa.area_max'"
    assert set(qa["gid"]) == set(SHEETS), "'qa.gid' phải chứa listing, review và area"
    assert qa["key_col"], "Thiếu 'qa.key_col' trong file cấu hình"
    for k in ("number_of_reviews", "missing_price"):  # YAML tự đổi khóa ngày thành kiểu date
        qa[k] = {str(d): n for d, n in qa[k].items()}
    assert set(qa["number_of_reviews"]) == set(snaps), "'qa.number_of_reviews' phải có đúng các snapshot"
    assert set(qa["missing_price"]) <= set(snaps), "'qa.missing_price' chứa snapshot không tồn tại"
    return config

def _path(config, key):
    v = config.get(key)
    assert isinstance(v, str) and v.strip(), f"Thiếu cấu hình '{key}'"
    return ROOT / v  # v tuyệt đối thì ROOT / v = v

def _dirs(config):
    raw = _path(config, "raw_dir")
    return raw / "phan_cong", _path(config, "processed_dir") / "flagged", raw

def _validate_columns(data, required, path):
    missing = required - set(data.columns)
    assert not missing, f"{path} thiếu các cột: {sorted(missing)}"

def download_sheet(name, gid, sheet_id, out_dir):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    def write(tmp):
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        n = 0
        with gzip.open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                n += len(chunk)
        assert n > 0, f"Google Sheets trả về dữ liệu rỗng: {name}"

    _atomic(out_dir / f"{name}.csv.gz", write)

def _load_assignments(config, d):
    print("Đang phân công nhiệm vụ!")
    qa = config["qa"]
    for n in SHEETS:
        if not (d / f"{n}.csv.gz").is_file():
            download_sheet(n, qa["gid"][n], qa["sheet_id"], d)
    listing, review = _read_csv(d / "listing.csv.gz"), _read_csv(d / "review.csv.gz")
    _validate_columns(listing, {"listing_id", "analysis_area"}, "listing.csv.gz")
    _validate_columns(review, {"review_id", "Mẫu LLM", "Nhãn tay"}, "review.csv.gz")
    for col, val in qa["key_col"].items():
        _validate_columns(listing, {col}, "listing.csv.gz")
        _validate_columns(review, {col}, "review.csv.gz")
        listing = listing.loc[listing[col].astype(str).str.strip().eq(str(val).strip())].copy()
        review = review.loc[review[col].astype(str).str.strip().eq(str(val).strip())].copy()
    for c in ("listing_id", "analysis_area"):
        listing[c] = listing[c].str.strip()
    review["review_id"] = review["review_id"].str.strip()
    for c in ("Mẫu LLM", "Nhãn tay"):
        review[c] = review[c].str.strip().str.lower()
    listing, review = listing.loc[listing["listing_id"].ne("")], review.loc[review["review_id"].ne("")]
    assert not (listing.empty or review.empty), "Không tìm thấy phân công hợp lệ theo qa.key_col"
    assert listing["listing_id"].nunique() == len(listing), "Phân công listings có listing_id bị trùng"
    assert review["review_id"].nunique() == len(review), "Phân công reviews có review_id bị trùng"
    assert len(listing) == sum(qa["room_types"].values()), "Số listing phân công khác tổng qa.room_types"
    llm, manual = review["Mẫu LLM"].eq("x"), review["Nhãn tay"].eq("x")
    assert llm[manual].all(), "Toàn bộ review có nhãn tay phải thuộc mẫu LLM"
    # kiểm tra số lượng review/mẫu LLM/nhãn tay nếu YAML khai báo qa.reviews / qa.llm_sample / qa.manual_sample
    for k, n in {"reviews": len(review), "llm_sample": llm.sum(), "manual_sample": manual.sum()}.items():
        assert qa.get(k, n) == n, f"Phân công có {n} {k}, cấu hình yêu cầu {qa[k]}"
    return listing, review

def _filter_ids(src, out, ids, label, areas=None):
    df = _read_csv(src)
    _validate_columns(df, {"id"}, src)
    df["id"] = df["id"].str.strip()
    df = df.loc[df["id"].isin(ids)].copy()
    assert set(df["id"]) == ids and len(df) == len(ids), f"{src} không khớp đúng {len(ids)} {label} được phân công"
    if areas is not None:
        df["analysis_area"] = df["id"].map(areas)
    _write_csv(df, out)

def _filter_calendar(src, out, ids):
    cols, expected = ["listing_id", "date", "available"], len(ids) * CAL_DAYS
    _validate_columns(pd.read_csv(src, compression="gzip", nrows=0), set(cols), src)

    def write(tmp):
        n = 0
        with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
            for c in pd.read_csv(src, dtype=str, keep_default_na=False, compression="gzip", chunksize=100_000, usecols=cols):
                c["listing_id"] = c["listing_id"].str.strip()
                c = c.loc[c["listing_id"].isin(ids)]
                if not c.empty:
                    c.to_csv(f, index=False, header=n == 0)
                    n += len(c)
        assert n > 0, f"Không tìm thấy dữ liệu phù hợp để ghi: {out}"
        assert n == expected, f"{out} có {n} dòng, cần {expected}"

    _atomic(out, write)

def phan_cong(config, la, ra):
    adir, _, raw = _dirs(config)
    lids, rids = set(la["listing_id"]), set(ra["review_id"])
    areas = la.set_index("listing_id")["analysis_area"]
    for s in _snapshots(config):
        src, out = raw / s, adir / s
        _filter_ids(src / "listings.csv.gz", out / "listings.csv.gz", lids, "listing", areas)
        _filter_calendar(src / "calendar.csv.gz", out / "calendar.csv.gz", lids)
        _filter_ids(src / "reviews.csv.gz", out / "reviews.csv.gz", rids, "review")

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

def _read_area_bounds(config):
    path = _dirs(config)[0] / "area.csv.gz"
    a = _read_csv(path)
    _validate_columns(a, {"Thành phố", "analysis_area", *AREA_COLS}, path)
    city = str(config["city"]).strip()
    a = a.loc[a["Thành phố"].str.strip().eq(city)].copy()
    assert not a.empty, f"Không tìm thấy khu vực của thành phố {config['city']}"
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

def _self_check(qa, s, L, bounds):
    """Tự kiểm tra sau khi lọc (mục 3 của task.md); ngưỡng lấy từ qa trong YAML."""
    rt = L["room_type"].str.strip().value_counts().to_dict()
    assert rt == qa["room_types"], f"{s}: room_type {rt} khác cấu hình {qa['room_types']}"
    zero = int(_num(L["number_of_reviews"]).eq(0).sum())
    assert zero == qa["number_of_reviews"][s], f"{s}: có {zero} listing number_of_reviews = 0, cần {qa['number_of_reviews'][s]}"
    if s in qa["missing_price"]:
        n = int(_missing(L["price"]).sum())
        assert n == qa["missing_price"][s], f"{s}: có {n} listing thiếu price, cần {qa['missing_price'][s]}"
    c = L["analysis_area"].str.strip().value_counts()
    assert set(c.index) == set(bounds.index) and c.between(qa["area_min"], qa["area_max"]).all(), \
        f"{s}: vùng không khớp sheet area hoặc số listing/vùng ngoài [{qa['area_min']}, {qa['area_max']}]: {c.to_dict()}"

def _build_flag_frames(config):
    adir, qa, snaps = _dirs(config)[0], config["qa"], _snapshots(config)
    inputs = {}
    for s in snaps:
        inputs[s] = {}
        for t, keys in TABLES.items():
            p = adir / s / f"{t}.csv.gz"
            inputs[s][t] = d = _read_csv(p)
            _validate_columns(d, REQUIRED[t] | set(keys), p)
    bounds = _read_area_bounds(config)
    ids = pd.concat([t["reviews"]["id"].drop_duplicates() for t in inputs.values()])
    review_cnt = ids[ids.str.strip().ne("")].value_counts()

    frames = {}
    for s, t in inputs.items():
        print(f"Đang gắn cờ mốc {s}!")
        L, C, R = t["listings"], t["calendar"], t["reviews"]
        _self_check(qa, s, L, bounds)
        use_price = s == max(snaps)  # chỉ dùng price ở mốc cuối
        price = _num(L["price"])
        room = L["room_type"].astype("string").str.strip()
        nrev = L["number_of_reviews"].astype("string").str.strip()
        has_info = pd.concat([~_missing(L[c]) for c in ["last_review", *L.filter(regex="^review_scores_").columns]], axis=1).any(axis=1)
        # không có giá hợp lệ: (0, 0) vẫn cho outlier = False vì mọi phép so sánh với NA đều bị fillna(False)
        q1, q3 = price.dropna().quantile([0.25, 0.75]) if price.notna().any() else (0, 0)
        iqr = q3 - q1
        frames[s, "listings"] = _flag(L, ["id"], {
            "missing_neigh_cleansed": _missing(L["neighbourhood_cleansed"]),
            "missing_room_type": _missing(L["room_type"]),
            "missing_number_of_reviews": _missing(L["number_of_reviews"]),
            "missing_price": _missing(L["price"]) & use_price,
            "invalid_reviews_num": ~_missing(L["number_of_reviews"]) & ~nrev.str.fullmatch(r"\d+", na=False),
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
            "calendar_days_per_listing_mismatch": g["date"].transform("size").ne(CAL_DAYS) | g["date"].transform("nunique").ne(CAL_DAYS),
        })
        frames[s, "reviews"] = _flag(R, ["id"], {
            "missing_review_date": _missing(R["date"]),
            "missing_comments": _missing(R["comments"]),
            "invalid_review_date": ~_missing(R["date"]) & pd.to_datetime(R["date"], errors="coerce").isna(),
            "dup_review_id_in_snapshot": R["id"].duplicated(keep=False),
            "dup_review_id_across_snapshots": R["id"].map(review_cnt).fillna(0).gt(1),
        })
    return frames

def run_qa(config):
    flagged_dir = _dirs(config)[1]
    for (s, t), flags in _build_flag_frames(config).items():
        _write_csv(flags, flagged_dir / s / f"{t}_flags.csv.gz")

def main():
    try:
        config = _read_config()
        la, ra = _load_assignments(config, _dirs(config)[0])
        phan_cong(config, la, ra)
        run_qa(config)
        print("Đã xong!")
        return 0
    except Exception as error:
        print(f"Lỗi: {type(error).__name__}: {error}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())