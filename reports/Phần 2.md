# Phần 2 – Đánh giá chất lượng dữ liệu 

## 1. Lỗi tải xuống và đọc dữ liệu
- Mốc: 2025-12-29.
- Lỗi: 1000 dòng **price** trống.

## 2. Số lượng review
- Review trong mẫu LLM: 1000
- Review được gán nhãn thủ công: 100

## 3. Các cột có từ 90% giá trị thiếu
|Mốc|Cột|
|---|---|
|**2025-09-27**|neighbourhood_group_cleansed; calendar_updated; license|
|**2025-12-29**|neighborhood_overview; host_since; host_response_time; host_response_rate; host_acceptance_rate; host_thumbnail_url; host_neighbourhood; host_total_listings_count; host_verifications; neighbourhood; neighbourhood_group_cleansed; price; calendar_updated; estimated_revenue_l365d; license; instant_bookable|
|**2026-03-30**|neighborhood_overview; host_since; host_response_time; host_response_rate; host_acceptance_rate; host_thumbnail_url; host_neighbourhood; host_total_listings_count; host_verifications; neighbourhood; neighbourhood_group_cleansed; calendar_updated; license; instant_bookable|
|**2026-06-29**|neighborhood_overview; host_since; host_response_time; host_response_rate; host_acceptance_rate; host_thumbnail_url; host_neighbourhood; host_total_listings_count; host_verifications; neighbourhood; neighbourhood_group_cleansed; calendar_updated; license; instant_bookable|

## 4. Các cột cần thiết bị thiếu và cách xử lý
- Cột **price** dùng cho KPI bị trống.
- Xử lý: thay mốc **2025-12-29** bằng mốc **2026-09-23**.

## 5. Kết luận
- Dữ liệu đủ điều kiện phân tích hay cần tải lại: mốc mới **2026-09-23** đủ để làm.
