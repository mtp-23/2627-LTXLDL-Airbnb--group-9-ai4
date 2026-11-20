# AI Usage — DSE2049-C04-G09-santiago

> Tài liệu này mô tả cách nhóm sử dụng AI (Gemini LLM) trong dự án phân tích đánh giá Airbnb Santiago, Chile.

---

## Thành viên phụ trách: Nguyễn Thanh Lộc — `src/llm.py`

### 1. Mô hình AI sử dụng

| Thuộc tính | Giá trị |
|---|---|
| Model | `gemini-3.5-flash-lite` |
| Provider | Google AI (Gemini API) |
| Chế độ gọi | Async batch, 20 review/request |
| Rate limit | 12 RPM (giới hạn an toàn dưới mức 15 RPM của free tier) |
| Nhiệt độ (temperature) | `0.0` (deterministic) |
| Structured output | JSON schema via Pydantic (`BatchReviewResponse`) |

---

### 2. Mục đích sử dụng

Gọi Gemini API để **phân tích cảm xúc và trích xuất khía cạnh** từ 1.000 review Airbnb được phân công cho nhóm (xác định qua `Mẫu LLM == x` trong `data/raw/phan_cong/review.csv.gz`).

Kết quả xuất ra `data/processed/llm_results.csv` gồm 4 trường:

| Cột | Mô tả |
|---|---|
| `review_id` | ID gốc của review |
| `sentiment` | `positive` / `negative` / `mixed` |
| `aspects_positive` | Các khía cạnh được khen (vd: `host, location`) |
| `aspects_negative` | Các khía cạnh bị chê (vd: `value, noise`) |
| `language` | Mã ngôn ngữ ISO 639-1 (vd: `en`, `es`, `pt`) |

---

### 3. Prompt được sử dụng

**Prompt duy nhất** được gửi lên Gemini cho mỗi batch 20 review (xem `src/llm.py`, hàm `process_batch_with_gemini`):

```
Bạn là chuyên gia phân tích dữ liệu Airbnb. Phân tích các đánh giá sau theo chuẩn:
1. overall_sentiment: 'positive' (khen/hài lòng), 'negative' (chê/trải nghiệm tệ), 'mixed' (cân bằng khen và chê).
2. aspects: Chỉ gán các khía cạnh thuộc 7 nhóm: location, cleanliness, host, value, amenities, noise, safety.
   Với mỗi khía cạnh, chỉ định rõ sentiment là 'positive' (khen) hoặc 'negative' (chê).
3. language: Mã ngôn ngữ ISO 639-1 (ví dụ: en, es, vi).
Giữ nguyên review_id.

Danh sách đánh giá:
[{"id": "...", "comments": "..."}, ...]
```

Không dùng system prompt riêng. Response schema được ràng buộc bằng Pydantic (`BatchReviewResponse`) và `response_mime_type="application/json"`.

---

### 4. Kết quả thực tế

| Chỉ số | Giá trị |
|---|---|
| Tổng review xử lý | 1.000 / 1.000 (100% phân công) |
| positive | 881 |
| mixed | 107 |
| negative | 12 |
| Khía cạnh phổ biến (positive) | `host`, `location`, `cleanliness` |
| Cache | `data/processed/llm_cache.jsonl` (1.000 dòng JSONL) |
| Output | `data/processed/llm_results.csv` (1.000 dòng) |

---

### 5. Kiểm soát chất lượng & xác minh

- **Lọc QA trước khi gọi LLM**: Loại review có cờ `missing_review_text` hoặc `dup_review_id_in_snapshot` từ `data/processed/flagged/`. Trong 1.000 review phân công, **0 review bị loại** vì cờ xấu.
- **Khớp 100% ID**: `llm_results.csv` chứa đúng 1.000 `review_id` khớp hoàn toàn với danh sách `Mẫu LLM == x` trong file phân công.
- **Cache & resume**: Nếu pipeline bị ngắt giữa chừng, lần chạy lại tự động bỏ qua các ID đã có trong cache (`llm_cache.jsonl`), không gọi API lại.
- **Xác minh offline**: Chạy `py -3.14 src/llm.py --skip-llm` để xuất CSV từ cache mà không cần API key.

---

### 6. Giới hạn & rủi ro đã xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Rate limit 429 | Retry tự động với exponential backoff (tối đa 5 lần, `tenacity`) |
| Mất kết nối giữa chừng | Cache JSONL ghi từng batch → resume tự động |
| Thiếu API key | Tự động dùng cache nếu có, hoặc báo lỗi rõ ràng |
| Review rỗng/NaN | 3 lớp lọc: QA flags → `dropna` → `str.strip().ne("")` |

---

### 7. Môi trường chạy

```
python: 3.14 (py -3.14)
google-genai: (Gemini SDK)
aiolimiter, tenacity, pydantic, pandas, pyyaml, python-dotenv
API key: GEMINI_API_KEY (đọc từ .env hoặc biến môi trường)
```

**Lệnh chạy:**
```bash
# Chạy đầy đủ (cần API key)
py -3.14 src/llm.py --config configs/santiago.yaml

# Chạy offline từ cache (không cần API key)
py -3.14 src/llm.py --skip-llm

# Ước tính chi phí
py -3.14 src/llm.py --estimate-cost 1000
```
