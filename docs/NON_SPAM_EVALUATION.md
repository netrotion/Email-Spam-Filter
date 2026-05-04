# Đánh Giá Dương Tính Giả Trên Thư Hợp Lệ

Mô hình: `contextual_deberta_v3_base`
Mục đích: đo tần suất thư hợp lệ bị gán nhầm là thư rác.

Dương tính giả là rủi ro quan trọng vì bộ lọc thư rác chặn nhầm thư hợp lệ sẽ không an toàn trong sử dụng thực tế.

## Thiết Lập Đo Kiểm

Mô hình hiện tại đã được huấn luyện với dữ liệu bổ sung từ SpamAssassin ham. Vì vậy phép đo này dùng các dòng SpamAssassin ham nằm trong tập kiểm thử đã xử lý, không dùng lại dòng thuộc tập huấn luyện.

Nguồn được xét:

- `spamassassin_easy_ham`
- `spamassassin_easy_ham_2`
- `spamassassin_hard_ham`

## Kết Quả Chỉ Trên Thư Hợp Lệ

| Nguồn | Số mẫu | Dự đoán ham | Dự đoán spam | Tỷ lệ dương tính giả | Xác suất spam trung bình |
| --- | ---: | ---: | ---: | ---: | ---: |
| `easy_ham` | 384 | 382 | 2 | 0.52% | 0.49% |
| `easy_ham_2` | 228 | 228 | 0 | 0.00% | 0.00% |
| `hard_ham` | 43 | 43 | 0 | 0.00% | 0.05% |
| Toàn bộ thư hợp lệ giữ lại | 655 | 653 | 2 | 0.31% | - |

## Khảo Sát Ngưỡng

| Ngưỡng | Độ chính xác kiểm thử | Độ chính xác dương tính (precision) | Độ nhạy (recall) | F1 kiểm thử | Tỷ lệ dương tính giả (FPR) trên ham |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 99.32% | 98.92% | 99.42% | 99.17% | 0.31% |
| 0.60 | 99.30% | 98.92% | 99.38% | 99.15% | 0.31% |
| 0.70 | 99.30% | 98.96% | 99.33% | 99.15% | 0.31% |
| 0.80 | 99.32% | 99.00% | 99.33% | 99.17% | 0.31% |
| 0.90 | 99.37% | 99.17% | 99.29% | 99.23% | 0.15% |
| 0.95 | 99.34% | 99.17% | 99.21% | 99.19% | 0.15% |

## Kết Luận Đo Kiểm

Quyết định: giữ mô hình hiện tại.

Lý do:

- F1 trên tập kiểm thử vẫn trên 99%.
- Tỷ lệ dương tính giả trên thư hợp lệ giữ lại thấp.
- Tăng ngưỡng lên 0.90 có thể giảm thêm FPR trên ham, nhưng ngưỡng mặc định 0.50 đã phù hợp cho minh họa trong bài tập lớn.

Ngưỡng vận hành khuyến nghị:

- Chế độ minh họa/mặc định: 0.50.
- Chế độ thận trọng nhằm giảm dương tính giả: 0.90.

## Ví Dụ Dương Tính Giả Đại Diện

Hai mẫu `easy_ham` bị gán nhầm là thư rác:

- Một email về mã nguồn/gỡ lỗi có mẫu văn bản bất thường.
- Một email dạng tin tức về chủ đề tôn giáo/văn hóa.

Không ghi nhận dương tính giả trong `easy_ham_2` hoặc `hard_ham` ở lần đo này.
