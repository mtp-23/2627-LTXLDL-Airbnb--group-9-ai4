# Phân tích thị trường Airbnb – Thành phố Santiago (Chile)

Dự án môn **Lập trình xử lý dữ liệu (`UET.DSE2049`)** — Lớp `UET.DSE2049 4`, Nhóm `9` (`DSE2049-C04-G09-santiago`).

- **Nguồn dữ liệu:** [Inside Airbnb](https://insideairbnb.com/) (giấy phép CC BY 4.0) — đường dẫn gốc `chile/rm/santiago`, tiền tệ `CLP`.
- **Bốn mốc chụp (snapshots):** `2025-09-27`, `2025-12-29`, `2026-03-30`, `2026-06-29`.
- **Phạm vi dữ liệu phân công:** 1.000 listing (875 `Entire home/apt`, 125 `Private room` theo 8 vùng phân tích `A01`–`A08`), 365.000 dòng lịch/mốc, 5.000 review/mốc (trong đó 1.000 review thuộc mẫu LLM và 100 review gán nhãn thủ công).
- **Bên đặt hàng phân tích:** Chính quyền thành phố Santiago.

---

## 1. Môi trường và cài đặt

Phiên bản Python đã kiểm thử được ghi trong tệp `.python-version` (`3.12.14`) và danh sách thư viện ghim phiên bản trong `requirements.txt`.

```bash
# 1. Tạo và kích hoạt môi trường ảo (Python 3.12.14)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Cấu hình khóa API Gemini (nếu chạy bước gọi LLM thật)
cp .env.example .env
# Điền khóa API vào biến GEMINI_API_KEY trong .env, hoặc export trực tiếp:
export GEMINI_API_KEY="your_gemini_api_key"
```


---

## 2. Chạy toàn bộ quy trình bằng một lệnh

Lệnh chuẩn chạy từ đầu đến cuối (tải dữ liệu → cắt theo phân công → kiểm tra QA → chạy LLM + Baseline → đánh giá trên 100 nhãn tay → tính 8 KPI → vẽ 3 biểu đồ):

```bash
python run_pipeline.py --config configs/santiago.yaml
```

Các tùy chọn hỗ trợ:

| Tham số | Ý nghĩa |
|---|---|
| `--config <path>` | Đường dẫn tệp cấu hình YAML (mặc định `configs/santiago.yaml`). |
| `--skip-download` | Bỏ qua tải lại dữ liệu thô nếu đã có sẵn trong `data/raw/`. |
| `--force` | Tải lại và ghi đè dữ liệu thô đã tồn tại. |
| `--skip-llm` | Không gọi Gemini API; dùng lại cache sẵn có tại `data/processed/llm_cache.jsonl`. |
| `--skip-evaluation` | Bỏ qua bước chấm điểm trên nhãn tay (`data/samples/hand_labeled.csv`). |
| `--snapshot <YYYY-MM-DD>` | Chọn mốc chụp để in và xuất bảng KPI 1–6 (mặc định là mốc mới nhất `2026-06-29`). |

---

## 3. Cấu trúc mã nguồn và phân vai mô-đun

Mỗi mô-đun trong `src/` chỉ đảm nhận đúng một nhiệm vụ xử lý thuần trong bộ nhớ; toàn bộ đọc cấu hình, I/O đĩa, cache và điều phối gom về `run_pipeline.py`:

```text
.
├── configs/
│   └── santiago.yaml                  # Cấu hình thành phố, mốc chụp, ngưỡng QA, tham số LLM
├── data/
│   ├── raw/                           # Dữ liệu thô tải về & bảng phân công (gitignore)
│   ├── processed/                     # Kết quả sau QA, KPI, LLM, Baseline, dữ liệu vẽ hình
│   └── samples/
│       ├── annotation_guidelines.md   # Quy chuẩn gán nhãn thủ công
│       └── hand_labeled.csv           # 100 review gán nhãn tay (Ground Truth)
├── figures/                           # 3 biểu đồ PNG sinh tự động
├── reports/
│   ├── check_DSE2049-C04-G09-santiago.csv
│   ├── Phần 2.md                      # Đánh giá chất lượng dữ liệu 4 mốc
│   └── qa_report.csv                  # Báo cáo tác động 20 quy tắc QA theo mốc
├── src/
│   ├── download.py                    # Tải và kiểm tra toàn vẹn CSV.gz, GeoJSON, Google Sheets
│   ├── assign.py                      # Cắt dữ liệu thô theo danh sách ID được phân công
│   ├── qa.py                          # Bộ 20 quy tắc QA gắn cờ và tổng hợp qa_report.csv
│   ├── llm.py                         # Gọi Gemini API (structured output, rate limit, retry) & ước tính chi phí
│   ├── baseline.py                    # Phương pháp đối chứng dùng regex (không LLM)
│   ├── evaluate.py                    # Chấm LLM & Baseline trên nhãn tay, phân tích lỗi
│   ├── kpi.py                         # Làm sạch theo cờ QA, tính 8 KPI & bảng theo khu vực/thời gian
│   └── viz.py                         # Vẽ 3 biểu đồ vào figures/
├── run_pipeline.py                    # Điểm vào duy nhất chạy toàn bộ quy trình
├── requirements.txt                   # Thư viện đã ghim phiên bản (==)
├── .python-version                    # 3.12.14
├── .env.example                       # Mẫu biến môi trường GEMINI_API_KEY
├── AI_USAGE.md                        # Nhật ký minh bạch sử dụng AI
└── README.md
```

---

## 4. Tệp kết quả để đối chiếu

Các bảng cờ QA lưu dạng `.csv.gz` để giảm dung lượng (`data/processed/` chỉ ~4.6 MB < 50 MB), còn các bảng KPI, đánh giá và báo cáo giữ nguyên dạng `.csv` chuẩn theo đề bài:

| Tệp đầu ra | Nội dung |
|---|---|
| `reports/qa_report.csv` | 80 dòng (4 mốc × 20 quy tắc QA): điều kiện, lý do, số dòng gắn cờ, KPI trước & sau QA. |
| `data/processed/flagged/<mốc>/*_flags.csv.gz` | Bảng cờ QA cấp từng dòng cho `listings`, `calendar`, `reviews` ở cả 4 mốc. |
| `data/processed/kpi_before_qa.csv` | Giá trị 8 KPI trước QA tại mốc `2026-06-29`. |
| `data/processed/kpi_after_qa.csv` | Giá trị 8 KPI sau QA tại mốc `2026-06-29` (kèm mục tiêu, công thức, đơn vị). |
| `data/processed/llm_cache.jsonl` | Kết quả thô dạng JSONL có cấu trúc cho 1.000 review mẫu LLM. |
| `data/processed/llm_results.csv` | Bảng phẳng 1.000 kết quả LLM (`review_id`, `sentiment`, `aspects_positive`, `aspects_negative`, `language`). |
| `data/processed/baseline_results.jsonl` | Kết quả phân loại bằng regex baseline trên 5.000 review phân công sạch. |
| `data/processed/evaluation_metrics.csv` | Bảng so sánh LLM vs Regex Baseline trên 100 nhãn tay. |
| `data/processed/llm_error_analysis.csv` | 28 trường hợp LLM lệch cảm xúc hoặc khía cạnh so với nhãn tay. |
| `data/processed/viz/fig1_areas.csv` | Số liệu KPI theo từng khu vực (`neighbourhood_cleansed`) ở mốc cuối. |
| `data/processed/viz/fig2_over_time.csv` | Số liệu KPI 3, 4, 5 qua 4 mốc thời gian. |
| `data/processed/viz/fig3_llm.csv` | Ma trận nhầm lẫn và F1 từng lớp của LLM trên 100 nhãn tay. |
| `figures/fig1_spatial_map.png` | Bản đồ phân bố listing và tỷ lệ ngày còn trống theo khu vực. |
| `figures/fig2_reviews_availability_over_time.png` | Xu hướng theo thời gian qua 4 mốc chụp. |
| `figures/fig3_llm_evaluation.png` | Ma trận nhầm lẫn và F1 từng lớp cảm xúc của LLM. |

---

## 5. Kết quả đối chiếu (Santiago — mốc `2026-06-29`)

### Bảng 8 KPI (Trước QA vs Sau QA)

| # | Tên KPI | Đơn vị | Trước QA | Sau QA | Ghi chú / Cơ sở tính (Sau QA) |
|---|---|---|---|---|---|
| 1 | Số lượng listing theo khu vực | Phòng | `387` | `387` | Quận `Santiago` (lớn nhất) trong `1,000` listing |
| 2 | Tỷ lệ nhà nguyên căn tại khu vực trọng điểm | `%` | `90.44` | `90.44` | Quận `Santiago`: `350 / 387` listing `Entire home/apt` |
| 3 | Tỷ lệ listing không có review trong 12 tháng qua | `%` | `13.20` | `13.20` | `132 / 1,000` listing (`number_of_reviews_ltm = 0`) |
| 4 | Số review trung bình trên listing | Review/listing | `81.96` | `81.96` | `81,960 / 1,000` listing |
| 5 | Tỷ lệ ngày có thể đặt phòng | `%` | `74.07` | `74.07` | `270,371 / 365,000` ngày `available = 't'` |
| 6 | Giá thuê phòng trung bình một đêm (mốc cuối) | `CLP/đêm` | `98,447.99` | `66,943.72` | Sau QA lọc `20` dòng `missing_price` và `88` ngoại lai IQR (`n=892`) |
| 7 | Tỷ lệ review tích cực theo LLM | `%` | `87.80` | `87.80` | `878 / 1,000` review mẫu LLM |
| 8 | F1-score của LLM | Điểm F1 (0–1) | `0.8186` | `0.8186` | Macro-F1 trên `100` review nhãn tay |

### Đánh giá LLM vs Regex Baseline (`n = 100` nhãn tay)

| Method | Sentiment Acc | Sentiment Macro-F1 | Aspect Micro-F1 | Aspect Macro-F1 | Language Acc |
|---|---:|---:|---:|---:|---:|
| **LLM (`gemini-3.5-flash-lite`)** | `0.970` | `0.819` | `0.940` | `0.950` | `1.000` |
| **Baseline (Regex)** | `0.930` | `0.565` | `0.740` | `0.767` | `1.000` |

- **Chi phí ước tính cho toàn thành phố (`690,112` review ở mốc `2026-06-29`, `batch_size = 20`):** `67,285,980` input tokens (`$20.1858`) + `55,208,960` output tokens (`$138.0224`) = **`$158.2082` USD**.
