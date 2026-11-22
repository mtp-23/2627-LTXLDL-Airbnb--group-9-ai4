# AI_USAGE.md

Khai báo minh bạch việc sử dụng công cụ AI trong bài tập lớn Inside Airbnb (Nhóm 9 – `UET.DSE2049 4`, thành phố Santiago).

## Công cụ đã dùng (vào việc gì)

| Công cụ | Dùng vào việc gì |
|---|---|
| **Gemini API (`gemini-3.5-flash-lite` qua `google-genai`)** | Phân tích cảm xúc (`positive` / `mixed` / `negative`), khía cạnh khen/chê (`location`, `cleanliness`, `host`, `value`, `amenities`, `noise`, `safety`) và ngôn ngữ cho 1.000 review trong `src/llm.py` (*bắt buộc theo đề bài*). |
| **OpenAI Codex** | Xây dựng `src/baseline.py` (regex đối chứng) và `src/evaluate.py` (đánh giá & phân tích lỗi). |
| **Claude** | Kiểm tra công thức và kết quả đầu ra KPI theo bảng công thức của nhóm (`src/kpi.py`), hỗ trợ trực quan hóa (`src/viz.py`), giải thích và gỡ lỗi môi trường (`numpy` bị Application Control chặn, `FileNotFoundError` ở `src/qa.py`). |
| **Trợ lý lập trình AI trong IDE (Antigravity — Claude / Gemini)** | Viết code `src/download.py` (tải dữ liệu thô & kiểm toàn vẹn) và `src/qa.py` (lọc phân công, gắn cờ lỗi & khảo sát 4 mốc); hỗ trợ viết code async & xử lý cache trong `src/llm.py`; đề xuất bộ 8 KPI phục vụ Chính quyền thành phố Santiago (`src/kpi.py`); tái cấu trúc phân tách mô-đun (`src/assign.py`, `run_pipeline.py`); rà soát toàn repo và sinh `reports/qa_report.csv`. |

---

## Prompt then chốt (5–10): bước nào, vào tệp nào

1. **Tải dữ liệu thô theo cấu hình** → `src/download.py`
   - *"Viết code cho `download.py` với vai trò tải toàn bộ nội dung của các file dữ liệu theo đúng đường dẫn trong file config, không sửa cấu trúc ngoài phạm vi. Chia code làm 4 phần: đọc cấu hình, tạo danh sách tải, tải file, `main`; bỏ các nhánh `if not` thừa cho những trường hợp không xảy ra."*
2. **Triển khai lọc phân công và gắn cờ QA (8 bước)** → `src/qa.py`
   - *"Hãy triển khai `src/qa.py` theo cấu hình `configs/santiago.yaml`. Đọc phân công listing/review và bounds khu vực từ Google Sheets, dữ liệu Airbnb từ `data/raw/<snapshot>/`, lọc theo nhóm được giao rồi lưu dữ liệu trung gian vào `data/raw/phan_cong/<snapshot>/`. Kiểm tra cấu hình, cột, số lượng phân công và các điều kiện QA đã nêu cho listings, calendar, reviews. Với từng snapshot, xuất cờ boolean vào `data/processed/flagged/<snapshot>/{listings,calendar,reviews}_flags.csv.gz`, giữ đúng khóa và tên cờ mà các module downstream đang sử dụng. Chỉ gắn cờ, không tự sửa hoặc xóa dòng lỗi. Xử lý lỗi rõ ràng, ghi file an toàn, giữ ID dạng chuỗi và kiểm tra đầu ra sau khi triển khai."* (theo 8 bước: (1) Đọc & kiểm tra YAML `configs/santiago.yaml` → (2) Tải bảng phân công Google Sheets nếu chưa có cache → (3) Lọc & kiểm tra phân công đúng lớp/nhóm → (4) Với mỗi snapshot, cắt listings/review phân công & calendar theo listing được giao → (5) Đọc & kiểm tra bảng ranh giới `area` → (6) Kiểm tra các điều kiện/số lượng mong đợi trong cấu hình → (7) Tạo bảng cờ lỗi cho listings, calendar, reviews → (8) Ghi bảng cờ ra `.csv.gz` qua tệp tạm rồi thay thế tệp đích để tránh ghi dở).
3. **Xây dựng luồng chạy LLM async tối ưu rate limit & cache** → `src/llm.py`
   - *"Viết mô-đun Python chạy async phân tích reviews Airbnb bằng google-genai SDK. Yêu cầu gom batch 20 review/request, khống chế AsyncLimiter 12 RPM (đảm bảo không vượt quá 15 RPM của gói Free), có retry lũy thừa tenacity khi gặp lỗi 429/50x, và ghi nối vào file cache llm_cache.jsonl."*
4. **Chuẩn hóa Structured Outputs (Pydantic) & nguồn dữ liệu phân công cho LLM** → `src/llm.py`
   - *Schema:* *"Tạo Pydantic Schema trích xuất đúng 7 khía cạnh theo annotation_guidelines: location, cleanliness, host, value, amenities, noise, safety; phân loại sentiment tổng thể (positive, mixed, negative) và gán nhãn khen/chê cho từng aspect kèm trích dẫn chứng minh."*
   - *Nguồn phân công:* *"Sửa lại hàm load review trong llm.py để đọc trực tiếp từ data/raw/phan_cong/review.csv.gz, lọc đúng mã DSE2049-C04-G09-santiago và cờ 'Mẫu LLM == x', đồng thời loại bỏ sạch 100% các review trống comment trước khi gọi API."*
5. **Xây dựng Baseline đối chứng bằng regex** → `src/baseline.py`
   - *"Read @AGENTS.md, @configs/.yaml, and @src/llm.py. Implement `src/baseline.py` using pure Python `re` (config-driven, zero hardcoding) to classify `language`, `aspects`, and `sentiment` (with negation handling) matching `src/llm.py` output schema."*
6. **Đánh giá & phân tích lỗi LLM vs Baseline** → `src/evaluate.py`
   - *"Read @AGENTS.md, @configs/.yaml, @src/baseline.py, @src/llm.py, and @data/samples/hand_labeled.csv. Implement `src/evaluate.py` to evaluate LLM and Baseline against 100 hand-labeled reviews using Accuracy/F1 (`MultiLabelBinarizer`) and export misclassified rows."*
7. **Kiểm tra công thức KPI & định dạng bảng kết quả terminal** → `src/kpi.py`
   - Kết hợp các prompt: *"Kiểm tra công thức kpi trong ảnh trên, nếu sai hãy giải thích vì sao?"*, *"Tại sao kpi trước và sau qa đều cùng một giá trị?"*, và *"in bảng kết quả ra terminal, mỗi KPI một giá trị"* (bảng terminal tự co theo bề rộng).
8. **Nâng cấp bộ 8 KPI phục vụ Chính quyền thành phố** → `src/kpi.py`, `src/qa.py`, `src/viz.py`
   - *"Khi chạy trên dữ liệu phân công, tỷ lệ `Entire home/apt` (87.5%) và `number_of_reviews = 0` (25 ở mốc cuối) bị khóa cứng. Hãy đề xuất bộ 8 KPI phục vụ chính quyền thành phố có đủ KPI theo khu vực, theo thời gian, dùng LLM và KPI về `price` ở mốc cuối để thể hiện tác động QA trước/sau."*
9. **Tái cấu trúc kiến trúc đơn trách nhiệm** → `run_pipeline.py`, `src/*.py`
   - *"Đọc toàn bộ repo, loại bỏ các hàm đọc config/I/O trùng lặp giữa các mô-đun trong `src/`, tách mỗi mô-đun chỉ làm đúng một việc thuần trong bộ nhớ và chuyển toàn bộ đọc/ghi tệp, cache, điều phối về `run_pipeline.py`."*
10. **Review toàn repo, sinh `reports/qa_report.csv` & tinh chỉnh biểu đồ** → `src/qa.py`, `src/viz.py`, `requirements.txt`
    - *"Review toàn bộ repo đối chiếu với `project_guide.md`, bổ sung hàm xuất `reports/qa_report.csv`, sửa lỗi chồng nhãn trên biểu đồ, dọn gọn `requirements.txt` cross-platform và kiểm tra lại toàn bộ pipeline."*

---

## AI sai ở đâu (ví dụ thật)

### A. Lỗi trong mã nguồn, cấu hình & trực quan hóa đã phát hiện và sửa
*(Ghi nhận: `src/baseline.py` và `src/evaluate.py` khi sinh lần đầu từ OpenAI Codex không phát sinh lỗi logic trong kiểm thử 10 mẫu; tuy nhiên quá trình tích hợp và các bước khác gặp các lỗi thật sau đây)*:

1. **Sinh nhiều nhánh `if not` thừa và lỗi tải ban đầu (`src/download.py`):**
   - AI viết quá nhiều nhánh kiểm tra `if not` cho các trường hợp không bao giờ xảy ra khiến code dài và lan man. Ở lần chạy đầu tiên, quá trình tải bị lỗi và hiện thông báo không tải được nên nhóm phải chỉnh sửa lại logic tải.
2. **Bỏ qua giới hạn dung lượng, viết code dài dòng và đề xuất luật QA thừa (`src/qa.py`):**
   - Dù đã chụp ảnh yêu cầu giới hạn dung lượng bài tập lên, AI ban đầu không để ý và xuất đầu ra dạng `.csv` thường rất tốn bộ nhớ; nhóm phải viết prompt nhắc lại để chuyển sang định dạng nén `.csv.gz`.
   - Khi viết hàm gắn cờ, AI viết nhiều nhánh `if not` lan man và dài dòng.
   - Khi tham khảo AI đề xuất bộ luật QA, AI sinh nhiều luật thừa luôn cho kết quả toàn `True` hoặc toàn `False` (ví dụ: kiểm tra hợp lệ của cột nhãn LLM và nhãn tay, vốn tải trực tiếp từ Google Sheets phân công đã chuẩn hóa sẵn).
3. **Tự chế tên khía cạnh & gọi nhầm đường dẫn cũ (`src/llm.py`):**
   - Khi mới sinh code, AI tự chế thêm khía cạnh không có trong chuẩn như `'room_quality'` và tự ý đổi `'host'` thành `'host_service'`.
   - Từng gọi nhầm đường dẫn cũ `data/samples/review.csv` (file đã bị xóa) thay vì `data/raw/phan_cong/review.csv.gz`.
4. **Biểu đồ bị chồng chữ (`src/viz.py`):**
   - Trong `fig3_llm_evaluation.png`, chú thích đường `macro F1` từng cắt đè lên cột `negative` và legend/nhãn số liệu trên `fig2_reviews_availability_over_time.png` chạm vào đường vẽ (đã chỉnh lại đặt nhãn trên/dưới điểm và dùng hộp nền trắng `zorder`).
5. **Lệch tên cột khi refactor (`src/evaluate.py`):**
   - Khi đổi cột `"Model"` → `"Method"` trong bảng `metrics`, dòng lấy `llm_sentiment_f1` vẫn gọi `metrics.Model == "LLM"`, khiến `run_pipeline.py` dừng với lỗi `AttributeError: 'DataFrame' object has no attribute 'Model'`.
6. **Rò rỉ cache ngoài mẫu & tham số `None` (`run_pipeline.py`):**
   - `stage_llm` từng trả về toàn bộ bản ghi trong cache (`records`) thay vì chỉ tập mẫu được giao (`in_sample`), và truyền `batch_size=conf.get("batch_size")` có thể thành `None` nếu YAML thiếu khóa.
7. **Thổi phồng ước tính chi phí LLM (`src/llm.py`):**
   - Hàm `estimate_total_cost` ban đầu nhân 150 token system prompt cho **từng review đơn lẻ** thay vì tính theo từng batch (`batch_size=100`), làm chi phí ước tính cho 690.112 review bị đội từ `$156.966` lên `$184.605`.
8. **Baseline chạy nhầm trên toàn thành phố (bản `src/baseline.py` cũ):**
   - Bản trước tái cấu trúc đọc thẳng `data/raw/` (~795.000 dòng review) thay vì 5.000 review được phân công, tạo ra tệp `baseline_results.jsonl` nặng ~83 MB (vượt giới hạn 50 MB của repo) trước khi được sửa lại còn 545 KB.

### B. Lỗi phân loại thật của Gemini LLM trên tập 100 nhãn tay (`data/processed/llm_error_analysis.csv`)
Trong 100 review đối chiếu với nhãn thủ công, LLM đạt Sentiment Accuracy `0.980` (chỉ sai cảm xúc 2 review) nhưng có 47 review lệch tập khía cạnh:
1. **`review_id=359204067` (Sót yếu tố chê nhẹ khi câu có 2 vế đối lập):**
   - *Văn bản:* `"A bit tight but my guess is pretty typical. The rooftop access was nice for enjoying the mountains at sunset. Valentina was a very prompt communicator!"`
   - *Nhãn tay:* `mixed` (`host, amenities`) — *LLM:* `positive` (chỉ nhận diện `host`, bỏ qua chi tiết phàn nàn phòng hơi chật).
2. **`review_id=1342130397727675023` (Bỏ sót khía cạnh tiêu cực trong đoạn góp ý dài):**
   - *Văn bản:* Khen khu phố an toàn, yên tĩnh nhưng chê hệ thống khóa 4 mã phức tạp và thiếu nước/cà phê chào đón.
   - *Nhãn tay:* `mixed` (`location, host, amenities, noise, safety`) — *LLM:* `positive` (`location, noise, safety`).
3. **`review_id=541593665` (Đúng cảm xúc `mixed` nhưng trích xuất thiếu khía cạnh):**
   - *Văn bản:* Bài viết trình bày rõ ràng hai phần `The pros:` (view đẹp, chỗ làm việc, gần bar/nhà hàng) và `The cons:` (thiếu sạch sẽ, an ninh).
   - *Nhãn tay:* `mixed` (`location, amenities, cleanliness, host, safety`) — *LLM:* `mixed` (`cleanliness, safety, host`, bỏ sót `location` và `amenities`).
---

## Nhóm kiểm chứng thế nào

1. **Kiểm chứng `src/download.py`:** Chạy trực tiếp script tải dữ liệu trên terminal, theo dõi trạng thái từng mốc và kiểm tra thực tế các tệp `.csv.gz` / `.geojson` đã lưu trong `data/raw/`.
2. **Kiểm chứng lọc phân công & gắn cờ (`src/qa.py`):** Tạo thư mục `test/` chứa các tệp notebook `.ipynb` (như `test/test.ipynb`), tách phần lọc dữ liệu theo danh sách ID và từng nhóm luật gắn cờ thành từng block code riêng biệt để chạy thử và đối chiếu kết quả từng bước.
3. **Kiểm chứng `src/baseline.py`:** Chọn ngẫu nhiên 10 review từ tập mẫu, tự đọc văn bản và gán nhãn tay đối chiếu theo đúng các biểu thức regex đã đặt trong code, kiểm tra kết quả baseline xuất ra trùng khớp hoàn toàn với nhãn gán tay theo luật.
4. **Kiểm chứng `src/evaluate.py`:** Chạy `evaluate.py` trên kết quả 10 mẫu xuất từ `baseline.py`, tự tính tay lại từng chỉ số (`Accuracy`, `Macro-F1`, `Weighted-F1`,...) và đối chiếu xác nhận khớp 100% với đầu ra của `evaluate.py`.
5. **Kiểm chứng `src/llm.py`:** Rà soát code từng dòng, đối chiếu danh sách 7 aspect với `annotation_guidelines.md` và `hand_labeled.csv` của thành viên gán nhãn; chạy test thực tế với cờ `--skip-llm` để kiểm tra tệp xuất ra đủ chuẩn 1.000 dòng dữ liệu phân công.
6. **Kiểm chứng `src/viz.py`:** Mở xem trực tiếp từng tệp ảnh PNG sau khi chạy thử để phát hiện và xử lý lỗi chồng chữ, đường cắt nhãn.