# Báo Cáo Kỹ Thuật
## Hệ Thống Phân Loại Thư Rác Email Bằng Mô Hình Ngôn Ngữ Theo Ngữ Cảnh

Đề tài: BTL-TTNT - Phân loại thư rác email
Cập nhật: 2026-05-04
Trạng thái hiện tại: artifact mô hình DeBERTa-v3-base đã chạy ổn định khi ghim `transformers==5.0.0`

---

## 1. Tóm Tắt

Đề tài xây dựng hệ thống phân loại thư rác email hoàn chỉnh, bao gồm thu thập dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá, suy luận, giao diện web, API JSON và kiểm thử hồi quy.

Mô hình tốt nhất hiện tại là bộ phân loại Transformer theo ngữ cảnh:

| Nội dung | Giá trị |
| --- | --- |
| Mô hình tốt nhất | `contextual_deberta_v3_base` |
| Mô hình nền | `microsoft/deberta-v3-base` |
| Nhóm mô hình | Transformer theo ngữ cảnh |
| Lớp phân loại | `ham` = 0, `spam` = 1 |
| Chế độ huấn luyện | Tinh chỉnh toàn bộ mô hình nền |
| Tập huấn luyện / thẩm định / kiểm thử | 27,400 / 5,872 / 5,872 |
| Độ chính xác kiểm thử | 99.32% |
| Điểm F1 kiểm thử | 99.17% |
| ROC AUC kiểm thử | 99.96% |
| Tỷ lệ dương tính giả trên thư hợp lệ | 0.31% tại ngưỡng 0.50 |

Ghi chú vận hành quan trọng: bộ mã hóa đã được xuất bằng `transformers 5.0.0`. Khi cùng artifact mô hình bị nạp bằng `transformers 5.7.0`, xác suất dự đoán từng bị kéo về gần 50/50. Dự án hiện ghim `transformers==5.0.0` và bộ nạp mô hình sẽ báo lỗi sớm nếu phiên bản Transformers trong môi trường chạy không khớp.

## 2. Phát Biểu Bài Toán

Bài toán là phân loại nhị phân nội dung email:

- `ham`: thư hợp lệ.
- `spam`: thư rác, quảng cáo không mong muốn, lừa đảo, giả mạo hoặc có nguy cơ độc hại.

Kết quả suy luận gồm:

- nhãn dự đoán và mã nhãn
- xác suất thư rác và xác suất thư hợp lệ
- độ tin cậy
- điểm quyết định
- tên mô hình và siêu dữ liệu
- phần giải thích đặc trưng nếu dùng mô hình cổ điển

Tiêu chí thành công của hệ thống:

- phát hiện được phần lớn thư rác
- hạn chế dương tính giả trên thư hợp lệ
- trả xác suất ổn định
- hỗ trợ dự đoán đơn lẻ và dự đoán theo lô
- có siêu dữ liệu mô hình phục vụ báo cáo và gỡ lỗi
- có API và quy trình web có thể kiểm thử

## 3. Bộ Dữ Liệu

Quy trình huấn luyện kết hợp ba nhóm dữ liệu.

| Nguồn | Số dòng thô trước khi loại trùng/chia tập | Vai trò |
| --- | ---: | --- |
| Kaggle `venky73/spam-mails-dataset` | 5,171 | bộ dữ liệu email có nhãn spam/ham |
| Hugging Face / Enron spam corpus | 33,716 | kho email lớn |
| SpamAssassin ham corpora | 4,112 được đưa vào dữ liệu cuối | bổ sung thư hợp lệ |

Sau chuẩn hóa và loại trùng, thống kê cuối trong `models/spam_detector_metadata.json` là:

| Tập / lớp | Số lượng |
| --- | ---: |
| Tổng số mẫu | 39,144 |
| Mẫu thư rác | 16,010 |
| Mẫu thư hợp lệ | 23,134 |
| Tập huấn luyện | 27,400 |
| Tập thẩm định | 5,872 |
| Tập kiểm thử | 5,872 |
| Thư hợp lệ / thư rác trong tập kiểm thử | 3,470 / 2,402 |

Phân bố theo nguồn sau xử lý:

| Nguồn | Số lượng |
| --- | ---: |
| `enron_spam_setfit` | 30,138 |
| `kaggle_venky73_spam_mails` | 4,894 |
| `spamassassin_easy_ham` | 2,472 |
| `spamassassin_easy_ham_2` | 1,389 |
| `spamassassin_hard_ham` | 251 |

Nhóm thư hợp lệ SpamAssassin được bổ sung vì các mô hình ban đầu có xu hướng đánh dấu nhầm bản tin, email danh sách thư và email kỹ thuật là thư rác. Các nguồn `easy_ham`, `easy_ham_2` và `hard_ham` giúp mô hình học thêm những ví dụ hợp lệ nhưng có hình thức dễ bị nhầm với thư rác.

## 4. Chuẩn Bị Dữ Liệu

Các mô-đun chính:

- `src/data/download.py`: tải dữ liệu và dùng lại bộ đệm cục bộ
- `src/data/dataset.py`: chuẩn hóa nguồn, loại trùng và chia phân tầng
- `src/preprocessing/text.py`: làm sạch HTML và chuẩn hóa token

Quy trình:

1. Tải hoặc dùng lại dữ liệu thô trong `data/raw/`.
2. Chuẩn hóa từng nguồn về cấu trúc chung: `email_id`, `source`, `label`, `label_name`, `text`, `text_length`.
3. Gộp tiêu đề và nội dung thư khi nguồn dữ liệu có hai trường này.
4. Làm sạch văn bản:
   - loại HTML, script và style
   - thay URL bằng `URLTOKEN`
   - thay địa chỉ email bằng `EMAILTOKEN`
   - chuẩn hóa khoảng trắng
5. Loại trùng theo văn bản đã chuẩn hóa.
6. Chia phân tầng theo tỷ lệ 70/15/15 cho huấn luyện, thẩm định và kiểm thử.
7. Lưu các tệp CSV đã xử lý trong `data/processed/`.

Tiền xử lý vẫn giữ lại nhiều tín hiệu bề mặt vì phân loại thư rác thường phụ thuộc vào dấu câu, ký hiệu lặp, số, URL và mẫu địa chỉ email.

## 5. Kiến Trúc Mô Hình

Mô hình cuối là `ContextualTransformerSpamPipeline`, bao bọc mô hình phân loại chuỗi của Hugging Face.

`microsoft/deberta-v3-base` là mô hình ngôn ngữ tiền huấn luyện do Microsoft công bố. Đề tài không huấn luyện DeBERTa từ đầu, mà sử dụng mô hình này làm backbone, gắn đầu phân loại nhị phân và tinh chỉnh trên bộ dữ liệu email spam/ham của đề tài.

| Cấu hình | Giá trị |
| --- | --- |
| Mô hình nền | `microsoft/deberta-v3-base` |
| Độ dài tối đa | 256 token |
| Kích thước lô huấn luyện | 8 |
| Kích thước lô đánh giá | 16 |
| Số epoch | 3 |
| Tốc độ học | `1e-5` |
| Hệ số suy giảm trọng số | `0.01` |
| Tích lũy gradient | 2 bước |
| Tỷ lệ warmup của lịch học | 0.10 |
| Bộ mã hóa nền | được tinh chỉnh, không đóng băng |
| Hàm mất mát | cross entropy có trọng số theo lớp |
| Kiên nhẫn dừng sớm | 1 epoch |

DeBERTa là mô hình mã hóa, phù hợp với phân loại vì sinh ra điểm logit ổn định cho các nhãn cố định. Hệ thống không dùng mô hình sinh hội thoại để suy luận.

## 6. Quy Trình Huấn Luyện

Lệnh huấn luyện chính:

```bash
python -m src.models.train --mode contextual
```

Notebook `notebooks/train_on_google_colab.ipynb` dùng cùng cấu hình huấn luyện cuối cùng và là phương án khuyến nghị khi cần GPU.

Trình tự huấn luyện:

1. Chuẩn bị dữ liệu từ Kaggle, Enron và SpamAssassin.
2. Khởi tạo `microsoft/deberta-v3-base` với đầu phân loại 2 nhãn.
3. Huấn luyện bằng AdamW, khởi động tốc độ học, tích lũy gradient, trọng số lớp và cắt chuẩn gradient.
4. Đánh giá điểm F1 trên tập thẩm định sau mỗi epoch.
5. Lưu checkpoint có F1 thẩm định tốt nhất.
6. Đánh giá trên tập kiểm thử giữ lại.
7. Lưu artifact mô hình:
   - `models/spam_detector.joblib`
   - `models/spam_detector_metadata.json`
   - `models/spam_contextual_encoder/`
   - `docs/TRAINING_RESULTS.md`

Tóm tắt log huấn luyện:

| Epoch | Mất mát | F1 thẩm định | Điểm lưu mô hình |
| --- | ---: | ---: | --- |
| 1 | 0.1682 | 0.9874 | cập nhật |
| 2 | 0.0403 | 0.9911 | cập nhật, tốt nhất |
| 3 | 0.0170 | 0.9904 | giữ checkpoint epoch 2 |

## 7. Đánh Giá

Chỉ số kiểm thử chính thức từ siêu dữ liệu đã lưu:

| Chỉ số | Giá trị |
| --- | ---: |
| Độ chính xác tổng thể (accuracy) | 0.9932 |
| Độ chính xác dương tính (precision) | 0.9892 |
| Độ nhạy (recall) | 0.9942 |
| Điểm F1 | 0.9917 |
| ROC AUC | 0.9996 |
| Điểm Brier | 0.0063 |
| Mất mát log | 0.0320 |

Ma trận nhầm lẫn:

| Thực tế / Dự đoán | Ham | Spam |
| --- | ---: | ---: |
| Ham | 3,444 | 26 |
| Spam | 14 | 2,388 |

Đo kiểm dương tính giả trên thư hợp lệ:

| Nguồn | Số mẫu | Tỷ lệ dương tính giả |
| --- | ---: | ---: |
| SpamAssassin `easy_ham` | 384 | 0.52% |
| SpamAssassin `easy_ham_2` | 228 | 0.00% |
| SpamAssassin `hard_ham` | 43 | 0.00% |
| Toàn bộ thư hợp lệ giữ lại | 655 | 0.31% |

Kiểm tra tương thích môi trường chạy:

- Với `transformers 5.7.0`, artifact mô hình từng sinh xác suất gần ngẫu nhiên 50/50.
- Với `transformers 5.0.0`, cùng artifact mô hình sinh dự đoán có độ tin cậy cao như kỳ vọng.
- Kiểm tra nhanh 64 dòng đầu của `data/processed/test.csv` sau khi sửa môi trường cho kết quả `accuracy=1.0`, `f1=1.0` và `log_loss=0.00046`. Đây chỉ là kiểm tra vận hành nhanh, không thay thế các chỉ số chính thức trên 5,872 mẫu kiểm thử.

## 8. Suy Luận Và API

Đường đi suy luận chính:

- `src/models/inference.py`
- `src/models/contextual.py`
- `src/api/services.py`
- `src/api/routes.py`

Phản hồi dự đoán gồm:

- `label`
- `label_id`
- `spam_probability`
- `ham_probability`
- `confidence`
- `decision_score`
- `top_signals`
- `model_name`

Mô hình Transformer không cung cấp trọng số đặc trưng kiểu TF-IDF thưa, nên `top_signals` được để trống có chủ đích đối với mô hình theo ngữ cảnh.

Các tuyến web và điểm cuối API:

| Điểm cuối | Phương thức | Chức năng |
| --- | --- | --- |
| `/` | GET | bảng điều khiển |
| `/predict` | POST | dự đoán đơn lẻ từ biểu mẫu |
| `/batch` | POST | dự đoán theo lô từ tệp CSV/TXT |
| `/model-info` | GET | trang siêu dữ liệu mô hình |
| `/api/health` | GET | trạng thái dịch vụ |
| `/api/model-info` | GET | siêu dữ liệu mô hình dạng JSON |
| `/api/predict` | POST | dự đoán đơn lẻ dạng JSON |
| `/api/batch-predict` | POST | dự đoán theo lô từ JSON hoặc tệp |
| `/api/dataset-samples` | GET | danh sách mẫu kiểm thử đã xử lý |
| `/api/dataset-sample/<email_id>` | GET | một mẫu kiểm thử đã xử lý |
| `/predict-sample` | POST | dự đoán một mẫu đã xử lý |
| `/api/raw-dataset-samples` | GET | danh sách mẫu thô Kaggle/Enron |
| `/predict-raw-sample` | POST | dự đoán một mẫu thô |

Giới hạn yêu cầu:

- kích thước tệp tải lên: mặc định 4 MiB (4,194,304 byte)
- số dòng dự đoán theo lô: 5,000 dòng
- độ dài văn bản: 20,000 ký tự/email

## 9. Ứng Dụng Web

Ứng dụng Flask cung cấp giao diện minh họa thực tế:

- dự đoán một email
- tải lên CSV/TXT để dự đoán theo lô
- xem thông tin mô hình và chỉ số đánh giá
- lấy mẫu từ tập kiểm thử đã xử lý
- lấy mẫu từ dữ liệu thô
- so sánh nhãn thật và nhãn dự đoán khi mẫu có nhãn

Chức năng lấy mẫu đã xử lý dùng `data/processed/test.csv`:

| Nội dung | Giá trị |
| --- | ---: |
| Số mẫu kiểm thử | 5,872 |
| Ham | 3,470 |
| Spam | 2,402 |

Chức năng lấy mẫu thô đọc các tệp cục bộ hiện tại:

| Tệp thô | Số dòng |
| --- | ---: |
| `data/raw/kaggle/spam_ham_dataset.csv` | 5,171 |
| `data/raw/enron/enron_spam.csv` | 33,716 |

Bộ lấy mẫu thô nạp tối đa 1,000 dòng mỗi nguồn vào bộ nhớ để hiển thị và kiểm tra ngẫu nhiên.

## 10. Kiểm Thử

Bộ kiểm thử pytest hiện thu thập 65 ca kiểm thử, bao phủ tiền xử lý, xử lý dữ liệu, suy luận mô hình, hành vi API, hành vi ứng dụng web, lấy mẫu dữ liệu đã xử lý, lấy mẫu dữ liệu thô và hồi quy môi trường chạy.

Lệnh kiểm tra đầy đủ sau khi dọn tài liệu và tạo bản sao lưu:

```bash
.venv\Scripts\python.exe -m pytest -q
```

Kết quả:

```text
65 passed
```

`pytest.ini` giới hạn việc thu thập kiểm thử trong `tests/` và bỏ qua `backups/`, `.tmp/`, `.venv/` để các bản sao lưu không xung đột với kiểm thử đang hoạt động. Lần chạy đầy đủ gần nhất ghi nhận `65 passed`; quá trình kiểm thử có thể in một số cảnh báo phụ thuộc và cảnh báo dọn dẹp Python từ `multiprocess.resource_tracker`, nhưng các cảnh báo này không làm thất bại kiểm thử.

Các kiểm thử hồi quy quan trọng:

- softmax của mô hình theo ngữ cảnh vẫn hữu hạn với logit cực trị
- đường dẫn bộ mã hóa xuất từ Colab được chuẩn hóa trên Windows
- bộ nạp artifact mô hình từ chối phiên bản Transformers không khớp
- API và đường đi dự đoán theo lô giữ đúng cấu trúc dữ liệu trả về

## 11. Hạn Chế

Các hạn chế đã biết:

- Dữ liệu huấn luyện chủ yếu là tiếng Anh.
- Mô hình có thể khái quát hóa với một số mẫu thư rác/thư hợp lệ tiếng Việt rõ ràng, nhưng chưa phải mô hình chuyên biệt cho tiếng Việt.
- Thư lừa đảo hiện đại và thư rác sinh bởi AI có thể lệch khỏi phân bố dữ liệu huấn luyện.
- Ứng dụng Flask là bản minh họa, chưa phải triển khai sản phẩm được gia cố đầy đủ.
- Hiện vật mô hình có dung lượng lớn vì bộ mã hóa DeBERTa được lưu cục bộ.
- Môi trường chạy phải dùng đúng phiên bản Transformers đã ghim cho artifact mô hình này.

Ghi chú về suy luận tiếng Việt: mô hình vẫn có thể dự đoán đúng một số đầu vào tiếng Việt vì URL, email, dấu câu, cụm từ khẩn cấp, mẫu tiền/thưởng và token từ con vẫn là tín hiệu hữu ích. Đây là khả năng khái quát hóa, không phải bằng chứng về độ phủ tiếng Việt vững chắc. Nếu triển khai thật cho tiếng Việt, cần bổ sung tập đánh giá tiếng Việt có nhãn và có thể tinh chỉnh thêm bằng mẫu tiếng Việt.

## 12. Tài Liệu Trong Kho Mã

Các tài liệu báo cáo đang hoạt động:

- `BAO_CAO_MUC_LUC.md`: mục lục và khung báo cáo học thuật tiếng Việt
- `REPORT.md`: báo cáo kỹ thuật chính
- `README.md`: tổng quan dự án và cách chạy nhanh
- `SETUP.md`: hướng dẫn cài đặt và vận hành
- `docs/TRAINING_RESULTS.md`: chỉ số huấn luyện và kiểm thử chính thức
- `docs/NON_SPAM_EVALUATION.md`: đánh giá tỷ lệ dương tính giả
- `docs/WEBAPP_TESTING.md`: ghi chú kiểm thử web/API
- `docs/DATASET_SAMPLING_FEATURE.md`: hướng dẫn lấy mẫu từ tập kiểm thử đã xử lý
- `docs/RAW_DATASET_SAMPLING.md`: hướng dẫn lấy mẫu từ dữ liệu thô

Các bản tóm tắt trùng lặp và ghi chú cũ về sự cố mô hình đã được chuyển vào `backups/markdown-archive-*`.

## 13. Kết Luận

Hệ thống cuối đáp ứng các mục tiêu chính của đề tài:

- có bộ phân loại thư rác theo ngữ cảnh đã huấn luyện
- đạt hiệu năng cao trên tập kiểm thử giữ lại
- có tỷ lệ dương tính giả thấp trên tập thư hợp lệ giữ lại
- có quy trình dữ liệu và artifact mô hình được tài liệu hóa
- có bảng điều khiển Flask và API JSON
- hỗ trợ lấy mẫu dữ liệu đã xử lý và dữ liệu thô để minh họa
- có kiểm thử hồi quy cho ổn định suy luận và tương thích phiên bản

Yêu cầu vận hành quan trọng nhất là giữ môi trường chạy khớp với artifact mô hình: cài phụ thuộc từ `requirements.txt`, đặc biệt là `transformers==5.0.0`.
