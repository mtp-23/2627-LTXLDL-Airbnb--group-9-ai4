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
- **Các cột phục vụ KPI 1–5 (`id`, `neighbourhood_cleansed`, `room_type`, `number_of_reviews`, `number_of_reviews_ltm`, `available`) và KPI 7–8 (`comments` trong mẫu LLM 1.000 review):** Không bị thiếu ở cả 4 mốc.
- **Cột `price` phục vụ KPI 6 (Giá thuê trung bình một đêm ở mốc cuối):** Ở mốc `2025-12-29`, 1.000/1.000 dòng bị thiếu `price` nên không sử dụng giá tại mốc này. Tại mốc cuối `2026-06-29`, có 20/1.000 dòng thiếu `price` và 88/1.000 dòng ngoại lai vượt ngưỡng IQR (`Q3 + 1.5 × IQR`); quy trình QA gắn cờ các dòng này (không xóa khỏi bảng dữ liệu chung) và loại khỏi mẫu tính KPI 6 sau QA (`n = 892` listing).

## 5. Kết luận
- Dữ liệu đủ điều kiện phân tích (sử dụng đầy đủ 4 mốc cho KPI 1–5, và chỉ dùng `price` tại mốc cuối `2026-06-29` cho KPI 6).
