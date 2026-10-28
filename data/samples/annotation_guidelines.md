# Quy tắc gán nhãn thủ công (Annotation Guidelines) - 100 Reviews

Tài liệu này quy định chuẩn gán nhãn Ground Truth cho 100 đánh giá thủ công, dùng để đánh giá mô hình LLM và Baseline trong bài tập lớn Inside Airbnb.

## 1. Cấu trúc nhãn (Schema)
- `review_id`: ID của bài đánh giá.
- `comments`: Nội dung bài đánh giá.
- `sentiment`: Cảm xúc tổng thể (`positive` | `mixed` | `negative`).
- `aspects`: Danh sách các khía cạnh được đề cập (phân cách bằng dấu phẩy).
- `language`: Mã ngôn ngữ ISO 2 ký tự (ví dụ: `en`, `es`, `vi`).

## 2. Quy tắc Sentiment
- **positive**: Toàn bộ bài viết khen ngợi hoặc hài lòng là chính. Quy tắc ưu tiên: Nếu có 1 chi tiết chê nhỏ mang tính góp ý nhẹ nhưng tổng thể trải nghiệm rất tốt --> Vẫn gán positive
- **negative**: Bài viết chê bai, phàn nàn hoặc trải nghiệm tồi tệ. Quy tắc ưu tiên: Nếu có 1 điểm khen nhỏ nhưng tổng thể khiếu nại nghiêm trọng --> Vẫn gán negative
- **mixed**: Có sự cân bằng rõ ràng giữa điểm khen và điểm chê.

## 3. Quy tắc Aspects
Các khía cạnh hợp lệ: `location`, `cleanliness`, `host`, `value`, `amenities`, `noise`, `safety`.
- Chỉ gán các khía cạnh thuộc danh sách trên.
- Nếu không đề cập khía cạnh nào trong 7 loại trên -> Để trống.
- Nếu khía cạnh được đề cập mang hướng tích cực thì thêm khía cạnh đó vào aspects_positive. Ngược lại, nếu khía cạnh đó mang hướng tiêu cực thì thêm khía cạnh đó vào aspects_negative
- Đánh giá có nhắc đến một tiện nghi hoặc vật dụng cụ thể được cung cấp (ban công, hồ bơi, bếp, điều hòa) không? → `amenities`
- Đánh giá có nói về khu vực xung quanh hoặc khu dân cư (kể cả tầm nhìn ra những nơi đó) không? → `location`
- Đánh giá có nói rằng nơi ở đáng đồng tiền không? → `value`
- Đánh giá có nói về mức độ sạch sẽ, gọn gàng hoặc được bảo trì tốt của căn hộ (bụi bẩn, mùi, rệp giường, ga giường thơm sạch, khăn bị ố) không? → `cleanliness`
- Đánh giá có nói về hành vi hoặc dịch vụ của một người (giao tiếp, hỗ trợ nhận phòng, phản hồi nhanh, thân thiện, linh hoạt) không? → `host`
- Đánh giá có nói về âm thanh hoặc sự yên tĩnh (tiếng xe cộ, hàng xóm, tường mỏng, công trình xây dựng, hoạt động về đêm, hoặc “yên bình”) không? → `noise`
- Đánh giá có nói về cảm giác an toàn hoặc mất an toàn (khu vực vào ban đêm, ổ khóa, tòa nhà có cổng bảo vệ, trộm cắp, lối vào đáng ngờ, hoặc “đi bộ về thấy an toàn”) không? → `safety`
## 4. Xử lý trường hợp đặc biệt (Edge Cases)
- Câu mỉa mai: Gán theo ý định thật của người viết.
- Icon/Emoji: Gán sentiment tương ứng, aspects để rỗng.