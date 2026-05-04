# Kết Quả Huấn Luyện

Mô hình: `contextual_deberta_v3_base`
Mô hình nền: `microsoft/deberta-v3-base`
Chế độ huấn luyện: tinh chỉnh toàn bộ mô hình theo ngữ cảnh
Thời điểm xuất artifact mô hình: `2026-05-03T22:37:58.507432+00:00`
Ghi chú môi trường chạy: cài `transformers==5.0.0` trước khi suy luận.

## Tóm Tắt Dữ Liệu

| Nội dung | Số lượng |
| --- | ---: |
| Tổng số mẫu | 39,144 |
| Mẫu thư rác | 16,010 |
| Mẫu thư hợp lệ | 23,134 |
| Tập huấn luyện | 27,400 |
| Tập thẩm định | 5,872 |
| Tập kiểm thử | 5,872 |

## Cấu Hình Huấn Luyện

| Tham số | Giá trị |
| --- | --- |
| Độ dài tối đa | 256 |
| Kích thước lô huấn luyện | 8 |
| Kích thước lô đánh giá | 16 |
| Số epoch | 3 |
| Tốc độ học | `1e-5` |
| Hệ số suy giảm trọng số | `0.01` |
| Tỷ lệ warmup của lịch học | `0.10` |
| Tích lũy gradient | 2 |
| Dừng sớm | 1 epoch |
| Trọng số lớp | bật |
| Đóng băng mô hình nền | không |

## Chỉ Số Thẩm Định

| Chỉ số | Giá trị |
| --- | ---: |
| Độ chính xác tổng thể (accuracy) | 0.9927 |
| Độ chính xác dương tính (precision) | 0.9900 |
| Độ nhạy (recall) | 0.9921 |
| Điểm F1 | 0.9911 |
| ROC AUC | 0.9997 |
| Điểm Brier | 0.0066 |
| Mất mát log | 0.0294 |

Ma trận nhầm lẫn trên tập thẩm định:

| Thực tế / Dự đoán | Ham | Spam |
| --- | ---: | ---: |
| Ham | 3,446 | 24 |
| Spam | 19 | 2,383 |

## Chỉ Số Kiểm Thử

| Chỉ số | Giá trị |
| --- | ---: |
| Độ chính xác tổng thể (accuracy) | 0.9932 |
| Độ chính xác dương tính (precision) | 0.9892 |
| Độ nhạy (recall) | 0.9942 |
| Điểm F1 | 0.9917 |
| ROC AUC | 0.9996 |
| Điểm Brier | 0.0063 |
| Mất mát log | 0.0320 |

Ma trận nhầm lẫn trên tập kiểm thử:

| Thực tế / Dự đoán | Ham | Spam |
| --- | ---: | ---: |
| Ham | 3,444 | 26 |
| Spam | 14 | 2,388 |

## Kiểm Tra Tương Thích Môi Trường Chạy

Sự cố phát hiện sau huấn luyện:

- Nạp artifact mô hình bằng `transformers 5.7.0` làm xác suất dự đoán tiến gần 0.5 trên các đầu vào không liên quan.
- Nạp cùng artifact bằng `transformers 5.0.0` cho dự đoán có độ tin cậy cao như kỳ vọng.

Cách khắc phục:

- `requirements.txt` ghim `transformers==5.0.0`.
- `ContextualTransformerSpamPipeline` kiểm tra phiên bản major/minor của Transformers trong artifact trước khi nạp mô hình.

Kiểm tra nhanh sau khi sửa:

| Kiểm tra | Kết quả |
| --- | --- |
| Hai ví dụ thủ công ham/spam | dự đoán đúng với độ tin cậy cao |
| 64 dòng đầu của tập kiểm thử | accuracy 1.0, F1 1.0, log loss 0.00046 |

Kiểm tra 64 dòng chỉ là kiểm tra nhanh môi trường chạy. Chỉ số kiểm thử chính thức vẫn là các chỉ số trên toàn bộ 5,872 mẫu ở trên.
