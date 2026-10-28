import pandas as pd
import argparse
import numpy as np
from pathlib import Path


# rules: name - condition - reason - todo (ưu tiên gắn cờ)
RULES = [
    # thiếu
    {
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
        "name": "invalid_human_label",
        "condition": "Nhãn thủ công không thuộc tập nhãn được quy định",
        "reason": "Không thể dùng nhãn này làm ground truth để tính F1-score",
        "todo": "Gắn cờ",
    },
    {
        "name": "invalid_listing_id",
        "condition": "id không thể chuyển thành định danh hợp lệ",
        "reason": "Có thể làm sai phép ghép dữ liệu giữa các bảng",
        "todo": "Gắn cờ",
    },
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
    {
        "name": "duplicate_review_id_in_evaluation",
        "condition": "Một review_id xuất hiện nhiều hơn một lần trong tập đánh giá LLM",
        "reason": "Review bị tính nhiều lần có thể làm sai kết quả đánh giá",
        "todo": "Gắn cờ và kiểm tra trước khi tính KPI",
    },
    # danh mục lệch
    {
        "name": "unexpected_listing_id",
        "condition": "Snapshot chứa ID không thuộc danh sách 1.000 ID chuẩn",
        "reason": "Có thể dữ liệu đã chứa listing ngoài phạm vi được phân công",
        "todo": "Gắn cờ",
    },
    {
        "name": "listing_id_set_mismatch_between_snapshots",
        "condition": "Tập ID listing giữa các snapshot không giống nhau",
        "reason": "Có thể làm sai lệch việc so sánh KPI theo thời gian",
        "todo": "Gắn cờ và báo cáo các ID khác biệt",
    },



    {
        "name": "calendar_row_count_mismatch",
        "condition": "Số dòng calendar khác 365.000 dòng trong một snapshot",
        "reason": "Có thể thiếu hoặc thừa ngày lịch so với 1.000 listing trong 365 ngày",
        "todo": "Gắn cờ và kiểm tra số dòng thực tế",
    },
    {
        "name": "calendar_days_per_listing_mismatch",
        "condition": "Một listing có số ngày calendar khác 365 ngày trong snapshot",
        "reason": "Dữ liệu lịch của listing có thể chưa đầy đủ hoặc bị thừa",
        "todo": "Gắn cờ",
    },



    {
        "name": "review_listing_id_not_in_listings",
        "condition": "listing_id của review không tồn tại trong tập ID listing được phân công",
        "reason": "Review có thể nằm ngoài phạm vi phân tích",
        "todo": "Gắn cờ và kiểm tra phạm vi dữ liệu",
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