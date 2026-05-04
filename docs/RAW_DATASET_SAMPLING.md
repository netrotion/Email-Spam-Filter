# Chức Năng Lấy Mẫu Từ Dữ Liệu Thô

Ứng dụng web cũng có thể lấy mẫu từ các tệp CSV Kaggle và Enron cục bộ trước bước chia huấn luyện/thẩm định/kiểm thử. Chức năng này phục vụ minh họa và kiểm tra thủ công, không dùng làm đánh giá chính thức.

## Tệp Dữ Liệu Thô Hiện Tại

| Nguồn thô | Tệp | Số dòng |
| --- | --- | ---: |
| Kaggle | `data/raw/kaggle/spam_ham_dataset.csv` | 5,171 |
| Enron | `data/raw/enron/enron_spam.csv` | 33,716 |

Bộ lấy mẫu nạp tối đa 1,000 dòng mỗi nguồn vào bộ nhớ. Khi có đủ hai nguồn, bảng điều khiển có thể lấy mẫu từ tối đa 2,000 ví dụ thô đã nạp.

## Chức Năng Trên Giao Diện

Phần lấy mẫu thô hỗ trợ:

- chọn mẫu thô ngẫu nhiên
- chọn mẫu thư rác thô ngẫu nhiên
- chọn mẫu thư hợp lệ thô ngẫu nhiên
- chọn mẫu Kaggle ngẫu nhiên
- chọn mẫu Enron ngẫu nhiên
- hiển thị nhãn thật
- hiển thị nguồn dữ liệu
- so sánh kết quả dự đoán

Mẫu thô hữu ích vì cho thấy đường đi suy luận sản xuất xử lý văn bản ít được chuẩn hóa như thế nào. Các mẫu này không nên dùng làm chỉ số cuối cùng vì có thể trùng ý niệm với dữ liệu huấn luyện và không phải tập kiểm thử giữ lại chính thức.

## Điểm Cuối API

### GET `/api/raw-dataset-samples`

Trả về các mẫu thô sẵn sàng hiển thị cùng thống kê số lượng.

Dạng phản hồi minh họa, trong đó số lượng có thể thay đổi theo lần lấy mẫu ngẫu nhiên:

```json
{
  "ok": true,
  "samples": [],
  "total": 2000,
  "spam_count": 1048,
  "ham_count": 952,
  "available_sources": ["kaggle", "enron"],
  "stats_by_source": {
    "kaggle_raw": {
      "total": 1000,
      "spam": 482,
      "ham": 518
    },
    "enron_raw": {
      "total": 1000,
      "spam": 566,
      "ham": 434
    }
  }
}
```

Số lượng lớp đã nạp có thể thay đổi vì các dòng được chọn ngẫu nhiên.

### POST `/predict-raw-sample`

Dự đoán một mẫu thô thông qua luồng bảng điều khiển HTML.

Trường biểu mẫu:

- `label_filter`: không bắt buộc, `0` cho ham hoặc `1` cho spam
- `source_filter`: không bắt buộc, `kaggle_raw` hoặc `enron_raw`

## Triển Khai

Tệp chính:

```text
src/api/raw_dataset_sampler.py
```

Tích hợp trong:

```text
src/api/routes.py
src/api/templates/index.html
```

Được kiểm thử bởi:

```text
tests/test_raw_dataset_sampling.py
```

## Các Bước Minh Họa

1. Đảm bảo tệp dữ liệu thô tồn tại:

```powershell
dir data\raw\kaggle\spam_ham_dataset.csv
dir data\raw\enron\enron_spam.csv
```

2. Khởi động ứng dụng:

```powershell
python -m src.api.app
```

3. Mở:

```text
http://127.0.0.1:5000
```

4. Dùng phần lấy mẫu từ dữ liệu thô.

## Ghi Chú

- Văn bản thô có thể nhiễu hơn văn bản đã xử lý.
- Dịch vụ suy luận vẫn áp dụng bước làm sạch đầu vào thông thường trước khi dự đoán.
- Bộ lấy mẫu thô cố ý giới hạn bộ nhớ sử dụng.
- Đánh giá chính thức nằm trong `docs/TRAINING_RESULTS.md` và `docs/NON_SPAM_EVALUATION.md`.
