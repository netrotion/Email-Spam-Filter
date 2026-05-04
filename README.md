# Dự Án Phân Loại Thư Rác Email

Hệ thống phân loại thư rác email hoàn chỉnh, gồm thu thập dữ liệu, tiền xử lý, huấn luyện Transformer theo ngữ cảnh, lưu artifact mô hình, bảng điều khiển Flask, API JSON, công cụ lấy mẫu dữ liệu, đánh giá và kiểm thử.

## Trạng Thái Hiện Tại

| Nội dung | Giá trị |
| --- | --- |
| Mô hình tốt nhất | `contextual_deberta_v3_base` |
| Mô hình nền | `microsoft/deberta-v3-base` |
| Độ chính xác kiểm thử | 99.32% |
| Điểm F1 kiểm thử | 99.17% |
| Tỷ lệ dương tính giả trên thư hợp lệ giữ lại | 0.31% |
| Yêu cầu môi trường chạy | `transformers==5.0.0` |

Mô hình đã huấn luyện có sẵn trong `models/`. Cài phụ thuộc từ `requirements.txt` trước khi suy luận. Artifact mô hình hiện tại được xuất bằng Transformers 5.0.0; phiên bản mới hơn không tương thích có thể tạo xác suất xấp xỉ 0.5 cho hai lớp.

## Chạy Nhanh

```powershell
cd "D:\LEVIETHUNG\SOURCE CODE\BTL-TTNT"
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.api.app
```

Mở:

```text
http://127.0.0.1:5000
```

## Chức Năng Chính

- Huấn luyện và đánh giá bộ phân loại thư rác DeBERTa theo ngữ cảnh.
- Dự đoán một email từ giao diện web hoặc API JSON.
- Dự đoán theo lô từ JSON, CSV hoặc TXT.
- Xem siêu dữ liệu mô hình và các chỉ số đánh giá.
- Kiểm tra nhanh bằng mẫu từ tập kiểm thử đã xử lý.
- Kiểm tra nhanh bằng mẫu thô Kaggle/Enron.
- Chạy kiểm thử hồi quy cho tiền xử lý, quy trình dữ liệu, suy luận mô hình, API và ứng dụng web.

## Cấu Trúc Dự Án

```text
src/
  api/                 ứng dụng Flask, dịch vụ, route, bộ lấy mẫu
  data/                tải và chuẩn bị dữ liệu
  evaluation/          chỉ số kiểm thử và đo dương tính giả
  models/              huấn luyện, suy luận, chuỗi xử lý Transformer
  preprocessing/       làm sạch văn bản và tạo đặc trưng phụ
data/
  raw/                 dữ liệu thô được lưu cục bộ
  processed/           CSV huấn luyện/thẩm định/kiểm thử
models/
  spam_detector.joblib
  spam_detector_metadata.json
  spam_contextual_encoder/
docs/
  TRAINING_RESULTS.md
  NON_SPAM_EVALUATION.md
  WEBAPP_TESTING.md
  DATASET_SAMPLING_FEATURE.md
  RAW_DATASET_SAMPLING.md
```

## Bộ Dữ Liệu

Bộ dữ liệu cuối sau xử lý có 39,144 mẫu:

| Tập / lớp | Số lượng |
| --- | ---: |
| Thư rác | 16,010 |
| Thư hợp lệ | 23,134 |
| Tập huấn luyện | 27,400 |
| Tập thẩm định | 5,872 |
| Tập kiểm thử | 5,872 |

Nguồn dữ liệu:

- Kaggle spam/ham dataset
- Enron spam corpus
- SpamAssassin `easy_ham`, `easy_ham_2`, `hard_ham`

## Huấn Luyện

Huấn luyện theo ngữ cảnh mặc định:

```powershell
python -m src.models.train --mode contextual
```

Khuyến nghị khi dùng GPU/Colab:

```powershell
python -m src.models.train --mode contextual `
  --model-name microsoft/deberta-v3-base `
  --max-length 256 `
  --batch-size 8 `
  --eval-batch-size 16 `
  --epochs 3 `
  --learning-rate 1e-5 `
  --gradient-accumulation-steps 2 `
  --early-stopping-patience 1 `
  --fine-tune-base-model
```

Notebook Colab:

```text
notebooks/train_on_google_colab.ipynb
```

## Đánh Giá

Đánh giá mô hình đã lưu:

```powershell
python -m src.evaluation.evaluate
```

Đo tỷ lệ dương tính giả trên thư hợp lệ:

```powershell
python -m src.evaluation.non_spam
```

Báo cáo:

- `docs/TRAINING_RESULTS.md`
- `docs/NON_SPAM_EVALUATION.md`

## Web/API

Khởi động máy chủ:

```powershell
python -m src.api.app
```

Các điểm cuối API chính:

| Điểm cuối | Phương thức | Chức năng |
| --- | --- | --- |
| `/api/health` | GET | kiểm tra trạng thái |
| `/api/model-info` | GET | siêu dữ liệu mô hình |
| `/api/predict` | POST | dự đoán đơn lẻ |
| `/api/batch-predict` | POST | dự đoán theo lô |
| `/api/dataset-samples` | GET | mẫu kiểm thử đã xử lý |
| `/api/dataset-sample/<email_id>` | GET | một mẫu kiểm thử đã xử lý |
| `/predict-sample` | POST | dự đoán mẫu đã xử lý từ bảng điều khiển |
| `/api/raw-dataset-samples` | GET | mẫu dữ liệu thô |
| `/predict-raw-sample` | POST | dự đoán mẫu thô từ bảng điều khiển |

Ví dụ:

```powershell
curl -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"WINNER claim your free prize now\"}"
```

## Kiểm Thử

Chạy toàn bộ kiểm thử:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Lần chạy đầy đủ gần nhất:

```text
65 passed
```

Kiểm thử hồi quy tập trung cho suy luận và môi trường chạy:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_runtime_regressions.py tests/test_pipeline.py -q
```

## Tài Liệu

Tài liệu chính:

- `BAO_CAO_MUC_LUC.md`: mục lục và khung báo cáo học thuật tiếng Việt
- `REPORT.md`: báo cáo kỹ thuật đầy đủ
- `SETUP.md`: hướng dẫn cài đặt và vận hành
- `docs/*.md`: các phụ lục chuyên đề

Các bản tóm tắt trùng lặp và ghi chú lỗi cũ đã được chuyển vào `backups/markdown-archive-*`.
