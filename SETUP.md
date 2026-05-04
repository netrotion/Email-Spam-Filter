# Hướng Dẫn Cài Đặt Và Vận Hành

Tài liệu này mô tả cách chạy cục bộ bộ phân loại thư rác đã huấn luyện.

## 1. Môi Trường

Khuyến nghị:

- Windows PowerShell
- Python 3.12
- Môi trường ảo sẵn có của dự án: `.venv`

Kích hoạt môi trường:

```powershell
cd "D:\LEVIETHUNG\SOURCE CODE\BTL-TTNT"
.venv\Scripts\activate
```

Cài phụ thuộc:

```powershell
pip install -r requirements.txt
```

Quan trọng: giữ `transformers==5.0.0`. Artifact mô hình DeBERTa hiện tại được xuất bằng 5.0.0. Khi nạp bằng 5.7.0, mô hình từng sinh xác suất gần 50/50.

Kiểm tra phiên bản:

```powershell
python -c "import transformers; print(transformers.__version__)"
```

Kỳ vọng:

```text
5.0.0
```

## 2. Artifact Và Dữ Liệu Bắt Buộc

Dự án cần các tệp/thư mục sau:

```text
models/spam_detector.joblib
models/spam_detector_metadata.json
models/spam_contextual_encoder/
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

Nếu thiếu, cần huấn luyện lại hoặc khôi phục từ bản sao lưu/bản xuất mô hình.

## 3. Chạy Ứng Dụng Web

```powershell
python -m src.api.app
```

Mở:

```text
http://127.0.0.1:5000
```

Biến môi trường tùy chọn:

```powershell
$env:FLASK_HOST="127.0.0.1"
$env:FLASK_PORT="5000"
$env:FLASK_DEBUG="0"
$env:SPAM_WEB_MAX_UPLOAD_BYTES="4194304"
```

## 4. Dùng API

Dự đoán đơn lẻ:

```powershell
curl -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"WINNER claim your free cash prize now\"}"
```

Dự đoán theo lô bằng JSON:

```powershell
curl -X POST http://127.0.0.1:5000/api/batch-predict `
  -H "Content-Type: application/json" `
  -d "{\"texts\":[\"meeting tomorrow\", \"free prize click now\"]}"
```

Kiểm tra trạng thái:

```powershell
curl http://127.0.0.1:5000/api/health
```

## 5. Đánh Giá Mô Hình

Chỉ số trên tập kiểm thử:

```powershell
python -m src.evaluation.evaluate
```

Đo tỷ lệ dương tính giả trên thư hợp lệ:

```powershell
python -m src.evaluation.non_spam
```

Báo cáo đầu ra:

```text
docs/TRAINING_RESULTS.md
docs/NON_SPAM_EVALUATION.md
```

## 6. Chạy Kiểm Thử

Kiểm thử hồi quy tập trung cho suy luận/API:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_runtime_regressions.py tests/test_pipeline.py -q
```

Toàn bộ kiểm thử:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Ghi chú: pytest có thể in cảnh báo dọn dẹp từ `multiprocess.resource_tracker` sau khi kiểm thử đã pass. Cảnh báo này không phải lỗi kiểm thử.

## 7. Huấn Luyện Lại

Chạy cục bộ hoặc trên GPU:

```powershell
python -m src.models.train --mode contextual
```

Khuyến nghị cho Colab/GPU:

1. Mở `notebooks/train_on_google_colab.ipynb`.
2. Dùng T4 GPU hoặc cấu hình mạnh hơn.
3. Chạy các ô cài đặt và chuẩn bị.
4. Chạy huấn luyện theo ngữ cảnh.
5. Tải/xuất `models/`, `docs/` và `data/processed/`.
6. Cài lại phụ thuộc từ `requirements.txt` trước khi suy luận cục bộ.

## 8. Xử Lý Sự Cố

### Xác suất dự đoán gần 50/50

Kiểm tra phiên bản Transformers:

```powershell
python -c "import transformers; print(transformers.__version__)"
```

Cách khắc phục:

```powershell
pip install -r requirements.txt
```

### Không nạp được mô hình

Kiểm tra đường dẫn artifact:

```powershell
dir models
dir models\spam_contextual_encoder
```

### Không thấy mẫu từ tập kiểm thử đã xử lý trên web

Kiểm tra:

```powershell
dir data\processed\test.csv
```

### Không thấy mẫu dữ liệu thô

Kiểm tra:

```powershell
dir data\raw\kaggle\spam_ham_dataset.csv
dir data\raw\enron\enron_spam.csv
```
