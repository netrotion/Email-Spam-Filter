# Kiểm Thử Ứng Dụng Web Và API

Tài liệu này tóm tắt cách kiểm thử ứng dụng Flask và API JSON.

## Phạm Vi Ứng Dụng

Ứng dụng web hỗ trợ:

- dự đoán một email
- dự đoán theo lô từ JSON, CSV hoặc TXT
- trang thông tin mô hình
- điểm cuối kiểm tra trạng thái mô hình
- lấy mẫu từ tập kiểm thử đã xử lý
- lấy mẫu từ dữ liệu thô Kaggle/Enron
- so sánh nhãn thật và nhãn dự đoán khi mẫu có nhãn

## Route Được Bao Phủ

| Điểm cuối | Phương thức | Chức năng |
| --- | --- | --- |
| `/` | GET | bảng điều khiển |
| `/predict` | POST | biểu mẫu dự đoán đơn lẻ |
| `/batch` | POST | dự đoán theo lô từ tệp tải lên |
| `/model-info` | GET | trang thông tin mô hình |
| `/api/health` | GET | kiểm tra trạng thái dạng JSON |
| `/api/model-info` | GET | siêu dữ liệu mô hình dạng JSON |
| `/api/predict` | POST | dự đoán đơn lẻ dạng JSON |
| `/api/batch-predict` | POST | dự đoán theo lô từ JSON/tệp |
| `/api/dataset-samples` | GET | danh sách mẫu đã xử lý |
| `/api/dataset-sample/<email_id>` | GET | một mẫu đã xử lý |
| `/predict-sample` | POST | dự đoán mẫu đã xử lý |
| `/api/raw-dataset-samples` | GET | danh sách mẫu thô |
| `/predict-raw-sample` | POST | dự đoán mẫu thô |

## Tệp Kiểm Thử

| Tệp kiểm thử | Trọng tâm |
| --- | --- |
| `tests/test_api.py` | hành vi API và route với dịch vụ giả lập |
| `tests/test_webapp_with_dataset.py` | hành vi web/API với tập kiểm thử đã xử lý thật |
| `tests/test_dataset_sampling.py` | bộ lấy mẫu từ tập kiểm thử đã xử lý |
| `tests/test_raw_dataset_sampling.py` | bộ lấy mẫu thô và route dữ liệu thô |
| `tests/test_runtime_regressions.py` | hồi quy môi trường chạy và bảo vệ nạp mô hình |

## Dữ Liệu Dùng Trong Kiểm Thử Web

Tập dữ liệu đã xử lý:

| Nội dung | Số lượng |
| --- | ---: |
| Tệp | `data/processed/test.csv` |
| Mẫu | 5,872 |
| Ham | 3,470 |
| Spam | 2,402 |

Tệp dữ liệu thô:

| Tệp | Số dòng |
| --- | ---: |
| `data/raw/kaggle/spam_ham_dataset.csv` | 5,171 |
| `data/raw/enron/enron_spam.csv` | 33,716 |

Bộ lấy mẫu thô nạp tối đa 1,000 dòng mỗi nguồn để hiển thị trên giao diện và kiểm tra ngẫu nhiên.

## Cách Chạy

Kiểm thử tập trung cho web/API:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_webapp_with_dataset.py tests/test_dataset_sampling.py tests/test_raw_dataset_sampling.py -q
```

Kiểm thử hồi quy môi trường chạy/suy luận:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_runtime_regressions.py tests/test_pipeline.py -q
```

Toàn bộ kiểm thử:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Kết quả đầy đủ gần nhất sau khi thêm `pytest.ini`:

```text
65 passed
```

`pytest.ini` giới hạn thu thập trong `tests/` và bỏ qua các bản sao trong `backups/`.

## Nội Dung Được Xác Minh

- Payload lỗi JSON có cấu trúc và mã trạng thái ổn định.
- Yêu cầu dự đoán theo lô rỗng được xử lý đúng.
- Tệp CSV/TXT tải lên được phân tích an toàn.
- Tệp tải lên quá lớn bị từ chối theo giới hạn cấu hình.
- Mẫu từ tập kiểm thử đã xử lý được nạp và hiển thị.
- Mẫu dữ liệu thô được nạp và lọc.
- Trang dự đoán mẫu có so sánh nhãn thật và nhãn dự đoán.
- Trang thông tin mô hình hiển thị đường dẫn artifact, nhãn, chỉ số và ghi chú.
- Bộ nạp mô hình theo ngữ cảnh kiểm tra phiên bản Transformers của artifact mô hình.

## Ghi Chú Kiểm Thử

Sau khi kiểm thử đã pass, Python có thể in:

```text
Exception ignored in: multiprocess.resource_tracker.__del__
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
```

Đây là cảnh báo dọn dẹp từ phụ thuộc và không làm lần chạy kiểm thử thất bại.
