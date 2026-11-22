"""Phân công: chọn đúng listing/review được giao cho nhóm từ dữ liệu thô.

Nhiệm vụ duy nhất: đọc bảng phân công (đã nạp thành DataFrame) và cắt dữ liệu
thô theo ID được giao. Không đọc/ghi tệp, không gắn cờ chất lượng (việc của qa.py).
"""
from __future__ import annotations

import pandas as pd

SHEETS = ("listing", "review", "area")
CALENDAR_COLUMNS = ["listing_id", "date", "available"]
RAW_REQUIRED = {"listings": {"id"}, "reviews": {"id"}, "calendar": set(CALENDAR_COLUMNS)}


def required_sheet_columns(qa: dict) -> dict:
    """Cột bắt buộc của từng sheet phân công (gồm cột khóa lọc nhóm)."""
    keys = set(qa["key_col"])
    return {"listing": {"listing_id", "analysis_area"} | keys,
            "review": {"review_id", "Mẫu LLM", "Nhãn tay"} | keys}


def parse_assignments(listing: pd.DataFrame, review: pd.DataFrame, qa: dict):
    """Lọc sheet theo qa.key_col, chuẩn hóa và kiểm tra số lượng được giao."""
    for col, val in qa["key_col"].items():
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
    # kiểm tra số lượng nếu YAML khai báo qa.reviews / qa.llm_sample / qa.manual_sample
    for k, n in {"reviews": len(review), "llm_sample": llm.sum(), "manual_sample": manual.sum()}.items():
        assert qa.get(k, n) == n, f"Phân công có {n} {k}, cấu hình yêu cầu {qa[k]}"
    return listing, review


def llm_sample_ids(review_assignment: pd.DataFrame) -> set[str]:
    """ID review thuộc mẫu LLM ('Mẫu LLM' = x)."""
    return set(review_assignment.loc[review_assignment["Mẫu LLM"].eq("x"), "review_id"])


def _subset(df: pd.DataFrame, ids: set[str], label: str) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].str.strip()
    df = df.loc[df["id"].isin(ids)].copy()
    assert set(df["id"]) == ids and len(df) == len(ids), f"Dữ liệu không khớp đúng {len(ids)} {label} được phân công"
    return df


def subset_listings(listings: pd.DataFrame, listing_assignment: pd.DataFrame) -> pd.DataFrame:
    """Listing được giao, kèm cột analysis_area."""
    ids = set(listing_assignment["listing_id"])
    out = _subset(listings, ids, "listing")
    out["analysis_area"] = out["id"].map(listing_assignment.set_index("listing_id")["analysis_area"])
    return out


def subset_reviews(reviews: pd.DataFrame, review_assignment: pd.DataFrame) -> pd.DataFrame:
    return _subset(reviews, set(review_assignment["review_id"]), "review")


def subset_calendar(chunks, listing_assignment: pd.DataFrame, days: int) -> pd.DataFrame:
    """Calendar của listing được giao, nhận các mảnh DataFrame (file lớn đọc theo khối)."""
    ids = set(listing_assignment["listing_id"])
    parts = []
    for chunk in chunks:
        chunk = chunk.copy()
        chunk["listing_id"] = chunk["listing_id"].str.strip()
        chunk = chunk.loc[chunk["listing_id"].isin(ids)]
        if not chunk.empty:
            parts.append(chunk[CALENDAR_COLUMNS])
    assert parts, "Không tìm thấy dữ liệu calendar phù hợp"
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == len(ids) * days, f"calendar có {len(out)} dòng, cần {len(ids) * days}"
    return out
