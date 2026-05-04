# Chức Năng Lấy Mẫu Từ Tập Kiểm Thử Đã Xử Lý

Ứng dụng web có thể kiểm tra trực tiếp mô hình đã huấn luyện trên các mẫu thuộc tập kiểm thử giữ lại. Chức năng này giúp minh họa nhanh vì người dùng không cần tự nhập hoặc tải email lên.

## Nguồn Dữ Liệu

Tệp:

```text
data/processed/test.csv
```

Chia tập hiện tại:

| Nội dung | Số lượng |
| --- | ---: |
| Tổng số mẫu | 5,872 |
| Ham | 3,470 |
| Spam | 2,402 |

Các cột bắt buộc:

- `email_id`
- `source`
- `label`
- `label_name`
- `text`
- `text_length`

## Chức Năng Trên Giao Diện

Trong bảng điều khiển, phần lấy mẫu đã xử lý cung cấp:

- 10 mẫu ngẫu nhiên để hiển thị
- nút chọn mẫu ngẫu nhiên
- nút chọn mẫu thư rác ngẫu nhiên
- nút chọn mẫu thư hợp lệ ngẫu nhiên
- nút kiểm tra theo từng dòng
- hiển thị nhãn thật
- hiển thị nguồn và độ dài văn bản
- so sánh đúng/sai sau khi dự đoán

## Điểm Cuối API

### GET `/api/dataset-samples`

Trả về các mẫu sẵn sàng hiển thị cùng thống kê số lượng.

Dạng phản hồi ví dụ:

```json
{
  "ok": true,
  "samples": [],
  "total": 5872,
  "spam_count": 2402,
  "ham_count": 3470
}
```

### GET `/api/dataset-sample/<email_id>`

Trả về một mẫu đã xử lý đầy đủ theo `email_id`.

### POST `/predict-sample`

Dự đoán một mẫu đã xử lý thông qua luồng bảng điều khiển HTML.

Trường biểu mẫu:

- `email_id`: mã mẫu cụ thể, không bắt buộc
- `label_filter`: không bắt buộc, `0` cho ham hoặc `1` cho spam

## Triển Khai

Tệp chính:

```text
src/api/dataset_sampler.py
```

Tích hợp trong:

```text
src/api/routes.py
src/api/templates/index.html
```

Được kiểm thử bởi:

```text
tests/test_dataset_sampling.py
tests/test_webapp_with_dataset.py
```

## Các Bước Minh Họa

1. Khởi động ứng dụng web:

```powershell
python -m src.api.app
```

2. Mở:

```text
http://127.0.0.1:5000
```

3. Dùng phần lấy mẫu từ tập kiểm thử đã xử lý.
4. Chọn mẫu ngẫu nhiên, thư rác ngẫu nhiên, thư hợp lệ ngẫu nhiên hoặc nút kiểm tra ở từng dòng.
5. Kiểm tra phần tóm tắt dự đoán và nhãn đúng/sai.

## Giá Trị Sử Dụng

Chức năng này hữu ích cho:

- minh họa nhanh trên lớp
- kiểm tra nhanh hành vi mô hình sau khi huấn luyện lại
- quan sát ví dụ thật trong tập kiểm thử giữ lại
- trình bày dự đoán đúng/sai cụ thể mà không cần chuẩn bị tệp tải lên
