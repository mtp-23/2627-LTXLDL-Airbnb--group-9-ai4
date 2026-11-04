import pandas as pd
import argparse
import numpy as np
from pathlib import Path
import os 

# Lấy thư mục gốc của dự án dựa trên vị trí file src/qa.py
project_root = Path(__file__).resolve().parent.parent

# Hoặc nếu chạy từ nhiều vị trí khác nhau mà muốn tìm linh hoạt:
def find_project_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data" / "in_sheet").is_dir():
            return p
    return Path(__file__).resolve().parent.parent

root = find_project_root()
project_root = root
TEAM_LISTINGS, TEAM_REVIEW, TEAM_VUNG = (root / "data" / "in_sheet" / f for f in ["Listing.csv", "Review.csv", "Vùng.csv"])
out = root / "data" / "loc_in_sheet"
out.mkdir(exist_ok=True)

filters = pd.Series(["UET.DSE2049 4", "9", "DSE2049-C04-G09-santiago"], index=["Lớp", "Nhóm", "Mã dữ liệu"])
select = lambda df: df.loc[df[filters.index].astype(str).eq(filters).all(axis=1)]

listing = select(pd.read_csv(TEAM_LISTINGS, usecols=[*filters.index, "listing_id", "analysis_area"], encoding="utf-8"))
listing[[*filters.index, "listing_id", "analysis_area"]].to_csv(out / "Listing.csv", index=False, encoding="utf-8")

revs = select(pd.read_csv(TEAM_REVIEW, usecols=[*filters.index, "review_id", "listing_id", "Mẫu LLM", "Nhãn tay"], encoding="utf-8"))
revs[[*filters.index, "review_id", "listing_id", "Mẫu LLM", "Nhãn tay"]].to_csv(out / "Review.csv", index=False, encoding="utf-8")

areas = pd.read_csv(TEAM_VUNG, index_col=0, encoding="utf-8")
areas.loc[["Santiago"]].to_csv(out / "Vùng.csv", encoding="utf-8")

data_dir = project_root / "data"
raw_dir = data_dir / "raw"
loc_dir = data_dir / "loc_in_sheet"
processed_dir = data_dir / "processed"

team_listings = pd.read_csv(
    loc_dir / "Listing.csv",
    usecols=["listing_id", "analysis_area"],
    dtype=str,
)
team_calendar = pd.read_csv(
    loc_dir / "Listing.csv",
    usecols=["Lớp", "Nhóm", "Mã dữ liệu", "listing_id"],
    dtype=str,
)
areas = pd.read_csv(loc_dir / "Vùng.csv", index_col=0, dtype=str)
revs = pd.read_csv(
    loc_dir / "Review.csv",
    usecols=["review_id", "Mẫu LLM", "Nhãn tay"],
    dtype=str,
)

for i in sorted(os.listdir(raw_dir)):
    output_dir = processed_dir / i
    os.makedirs(output_dir, exist_ok=True)

    tmp_listing = raw_dir / i / "listings.csv.gz"
    listings = pd.read_csv(tmp_listing, dtype=str)
    a = listings.merge(team_listings, left_on="id", right_on="listing_id", how="right")
    aa = a.merge(areas, on="analysis_area", how="left")
    aa = aa.drop(columns=["listing_id", "analysis_area"])
    aa.to_csv(output_dir / "listings.csv", index=False, encoding="utf-8")

    tmp_calendar = raw_dir / i / "calendar.csv.gz"
    calendar = pd.read_csv(tmp_calendar, dtype=str)
    bb = team_calendar.merge(calendar, on="listing_id", how="left")
    bb.to_csv(output_dir / "calendar.csv", index=False, encoding="utf-8")

    tmp_reviews = raw_dir / i / "reviews.csv.gz"
    reviews = pd.read_csv(tmp_reviews, dtype=str)
    c = reviews.merge(revs, left_on="id", right_on="review_id", how="right")
    cc = c.drop(columns=["review_id"])
    cc.to_csv(output_dir / "review.csv", index=False, encoding="utf-8")

# rules: name - condition - reason - todo (ưu tiên gắn cờ)
RULES = [
    # thiếu
    {
        # Cái này hơi thừa vì data không bị thiếu 
        "name": "missing_listing_id",
        "condition": "id bị thiếu/None/null",
        "reason": "Không thể định danh listing",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_neigh_cleansed",
        "condition": "neighbourhood_cleansed bị thiếu/None/null",
        "reason": "Không thể phân nhóm listing theo khu vực",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_room_type",
        "condition": "room_type bị thiếu/None/null",
        "reason": "Không thể xác định loại phòng để tính tỷ lệ",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_number_of_reviews",
        "condition": "number_of_reviews bị thiếu/None/null",
        "reason": "Không thể xác định listing có review hay không hoặc tính số review trung bình",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_price",
        "condition": "price bị thiếu/None/null",
        "reason": "Không thể sử dụng giá listing trong các phân tích cần đến giá",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_review_id",
        "condition": "id review bị thiếu/None/null",
        "reason": "Không thể định danh review để ghép dữ liệu và đánh giá nhãn",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_review_listing_id",
        "condition": "listing_id bị thiếu/None/null trong reviews",
        "reason": "Không thể xác định review thuộc listing nào",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_review_date",
        "condition": "date bị thiếu/None/null trong reviews",
        "reason": "Không thể xác định thời điểm review được tạo",
        "todo": "Gắn cờ",
    },
    {
        "name": "missing_review_text",
        "condition": "comments bị thiếu/None/null hoặc chỉ chứa khoảng trắng",
        "reason": "Không có nội dung để phân tích cảm xúc bằng LLM",
        "todo": "Gắn cờ và loại khỏi tập đầu vào LLM nếu không có nội dung",
    },
    {
        "name": "missing_listing_id_1000",
        "condition": "Một ID thuộc danh sách 1.000 ID chuẩn không xuất hiện trong snapshot",
        "reason": "Các snapshot không còn cùng phạm vi listing để so sánh",
        "todo": "Gắn cờ và báo cáo ID bị thiếu",
    },
    {
        "name": "missing_cal_list_id",
        "condition": "listing_id bị thiếu/None/null trong calendar",
        "reason": "Không thể xác định ngày lịch thuộc listing nào",
        "todo": "Gắn cờ",
    },

    # kiểu sai
    {
        "name": "invalid_reviews_num",
        "condition": "number_of_reviews không phải số nguyên hợp lệ",
        "reason": "Không thể tính chính xác số review và các KPI liên quan",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_room_type",
        "condition": "room_type không thuộc danh mục loại phòng được chấp nhận",
        "reason": "Có thể làm sai tỷ lệ các loại phòng",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_price",
        "condition": "price không thể chuyển thành giá trị số hợp lệ",
        "reason": "Không thể sử dụng giá trong các phép tính",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_calendar_date",
        "condition": "date không chuyển được sang kiểu ngày hợp lệ",
        "reason": "Không thể phân tích lịch theo thời gian",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_calendar_available",
        "condition": "available không thuộc tập giá trị hợp lệ của nguồn dữ liệu, thường là t/f",
        "reason": "Không thể tính chính xác tỷ lệ ngày còn trống",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_review_date",
        "condition": "date của review không chuyển được sang kiểu ngày hợp lệ",
        "reason": "Không thể phân tích review theo thời gian",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_llm_label",
        "condition": "Nhãn LLM không thuộc tập nhãn được quy định",
        "reason": "Nhãn không hợp lệ có thể làm sai KPI đánh giá mô hình",
        "todo": "Gắn cờ",
    },

    # ngoại lai
    {
        "name": "outlier_price",
        "condition": "price nằm ngoài ngưỡng IQR",
        "reason": "Giá bất thường có thể ảnh hưởng đến phân tích giá",
        "todo": "Gắn cờ",
    },
    # trùng
    {
        "name": "duplicate_listing_id",
        "condition": "id xuất hiện nhiều hơn một lần trong cùng snapshot listings",
        "reason": "Có thể làm tăng sai số lượng listing và ảnh hưởng các KPI",
        "todo": "Gắn cờ",
    },
    {
        "name": "duplicate_calendar_key",
        "condition": "Cặp (listing_id, date) xuất hiện nhiều hơn một lần trong cùng snapshot calendar",
        "reason": "Một listing có thể bị đếm nhiều lần trong cùng một ngày",
        "todo": "Gắn cờ",
    },
    {
        "name": "duplicate_review_id_within_snapshot",
        "condition": "ID review xuất hiện nhiều hơn một lần trong cùng snapshot reviews",
        "reason": "Có thể khiến một review bị đếm nhiều lần",
        "todo": "Gắn cờ",
    },
    # danh mục lệch
    {
        "name": "unexpected_listing_id",
        "condition": "Snapshot chứa ID không thuộc danh sách 1.000 ID chuẩn",
        "reason": "Có thể dữ liệu đã chứa listing ngoài phạm vi được phân công",
        "todo": "Gắn cờ",
    },
    {
        "name": "calendar_days_per_listing_mismatch",
        "condition": "Một listing có số ngày calendar khác 365 ngày trong snapshot",
        "reason": "Dữ liệu lịch của listing có thể chưa đầy đủ hoặc bị thừa",
        "todo": "Gắn cờ",
    },
    {
        "name": "review_id_overlap_between_snapshots",
        "condition": "Một ID review xuất hiện ở nhiều snapshot khác nhau",
        "reason": "Review có thể được giữ lại qua nhiều lần thu thập dữ liệu; cần phân biệt dữ liệu tích lũy với bản ghi trùng ngoài ý muốn",
        "todo": "Ghi nhận để kiểm tra; không tự động coi là lỗi hoặc xóa",
    },
    {
        "name": "unmatched_review_id_in_evaluation",
        "condition": "Không ghép được duy nhất review_id giữa kết quả LLM và nhãn thủ công",
        "reason": "Không thể đối chiếu dự đoán LLM với nhãn chuẩn một cách đáng tin cậy",
        "todo": "Gắn cờ và báo cáo ID không khớp",
    },
]