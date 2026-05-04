# Mục Lục Báo Cáo BTL-TTNT
## Hệ Thống Phân Loại Thư Rác Email Bằng Mô Hình Ngôn Ngữ Theo Ngữ Cảnh

Tài liệu này là khung mục lục học thuật cho báo cáo tiểu luận. Nội dung kỹ thuật chi tiết được triển khai trong `REPORT.md`; kết quả thực nghiệm, kiểm thử và hướng dẫn vận hành được đặt tại `docs/` và `SETUP.md`.

---

## Lời Nói Đầu

- Trình bày bối cảnh thư điện tử trong trao đổi học tập, công việc và dịch vụ trực tuyến.
- Nêu vấn đề thư rác: gây nhiễu thông tin, làm giảm hiệu quả xử lý hộp thư, tiềm ẩn rủi ro lừa đảo và có thể làm thất lạc thư hợp lệ nếu bộ lọc hoạt động không chính xác.
- Giới thiệu định hướng của đề tài: xây dựng hệ thống phân loại thư rác/thư hợp lệ dựa trên mô hình ngôn ngữ theo ngữ cảnh, có quy trình xử lý dữ liệu, mô hình học máy, ứng dụng web, API và kiểm thử.
- Nêu phạm vi báo cáo: tập trung vào bài toán phân loại nhị phân nội dung email, quy trình xây dựng dữ liệu, huấn luyện DeBERTa-v3-base, đánh giá mô hình và triển khai minh họa.

## Danh Mục Bảng, Hình Và Thuật Ngữ

- Danh mục bảng:
  - Bảng nguồn dữ liệu.
  - Bảng thống kê tập huấn luyện/thẩm định/kiểm thử.
  - Bảng cấu hình mô hình.
  - Bảng kết quả đánh giá.
  - Bảng tuyến web và điểm cuối API.
- Danh mục hình đề xuất:
  - Sơ đồ quy trình tổng thể.
  - Giao diện bảng điều khiển dự đoán.
  - Giao diện thông tin mô hình.
  - Ví dụ dự đoán đúng/sai từ mẫu dữ liệu.
- Thuật ngữ:
  - `ham`: thư hợp lệ.
  - `spam`: thư rác hoặc thư không mong muốn.
  - FPR: tỷ lệ dương tính giả, tức tỷ lệ thư hợp lệ bị gán nhầm là thư rác.
  - Precision: độ chính xác dương tính, đo tỷ lệ thư bị dự đoán là thư rác thật sự là thư rác.
  - Recall: độ nhạy, đo tỷ lệ thư rác thật sự được phát hiện đúng.
  - F1-score: điểm F1, trung bình điều hòa giữa precision và recall.
  - ROC AUC: diện tích dưới đường cong ROC, thể hiện khả năng phân biệt hai lớp trên nhiều ngưỡng.

## Chương 1. Mở Đầu

- 1.1. Lý do chọn đề tài
  - Email vẫn là kênh trao đổi phổ biến trong học tập và công việc.
  - Thư rác có thể chứa quảng cáo, lừa đảo, liên kết độc hại hoặc nội dung gây nhiễu.
  - Bài toán không chỉ là phát hiện thư rác, mà còn phải hạn chế việc chặn nhầm thư hợp lệ.
- 1.2. Dẫn chứng đi vào vấn đề
  - Bộ dữ liệu của đề tài gồm 39,144 email sau xử lý, trong đó có 16,010 thư rác và 23,134 thư hợp lệ.
  - Tập kiểm thử có 5,872 mẫu, gồm 3,470 thư hợp lệ và 2,402 thư rác.
  - Đánh giá riêng trên các mẫu thư hợp lệ SpamAssassin được giữ lại dùng để đo tỷ lệ dương tính giả, vì đây là rủi ro quan trọng trong bộ lọc thư rác.
  - Mô hình hiện tại đạt độ chính xác kiểm thử 99.32%, điểm F1 99.17% và tỷ lệ dương tính giả trên tập thư hợp lệ giữ lại là 0.31%.
- 1.3. Mục tiêu nghiên cứu
  - Xây dựng quy trình thu thập, chuẩn hóa và chia dữ liệu email.
  - Huấn luyện mô hình ngôn ngữ theo ngữ cảnh cho phân loại thư rác/thư hợp lệ.
  - Đánh giá mô hình bằng các chỉ số phân loại và đánh giá tỷ lệ dương tính giả.
  - Triển khai ứng dụng web/API để minh họa dự đoán đơn lẻ, dự đoán theo lô và dự đoán từ mẫu dữ liệu.
- 1.4. Đối tượng và phạm vi nghiên cứu
  - Đối tượng: nội dung email dạng văn bản.
  - Phạm vi: phân loại nhị phân `ham` và `spam`, tương ứng với thư hợp lệ và thư rác.
  - Dữ liệu chủ yếu là tiếng Anh; tiếng Việt chỉ được xem như khả năng khái quát hóa của mô hình, chưa phải trọng tâm huấn luyện.
- 1.5. Phương pháp thực hiện
  - Tổng hợp dữ liệu từ Kaggle, Enron và SpamAssassin.
  - Tiền xử lý HTML, URL, địa chỉ thư điện tử và khoảng trắng.
  - Tinh chỉnh `microsoft/deberta-v3-base`.
  - Đánh giá bằng độ chính xác tổng thể (accuracy), độ chính xác dương tính (precision), độ nhạy (recall), điểm F1 (F1-score), diện tích dưới đường cong ROC (ROC AUC), điểm Brier, mất mát log và ma trận nhầm lẫn.
  - Kiểm thử bằng pytest và minh họa qua Flask.
- 1.6. Bố cục báo cáo
  - Chương 1: Mở đầu.
  - Chương 2: Cơ sở lý thuyết và công nghệ sử dụng.
  - Chương 3: Phân tích và thiết kế hệ thống.
  - Chương 4: Xây dựng dữ liệu và tiền xử lý.
  - Chương 5: Huấn luyện mô hình.
  - Chương 6: Đánh giá thực nghiệm.
  - Chương 7: Xây dựng ứng dụng web và API.
  - Chương 8: Kiểm thử, thảo luận và kết luận.

## Chương 2. Cơ Sở Lý Thuyết Và Công Nghệ Sử Dụng

- 2.1. Bài toán phân loại văn bản
  - Khái niệm phân loại nhị phân.
  - Đặc điểm của dữ liệu email.
  - Ý nghĩa của xác suất dự đoán và ngưỡng phân loại.
- 2.2. Thư rác email và các đặc trưng thường gặp
  - URL, địa chỉ thư điện tử, lời kêu gọi hành động, nội dung quảng cáo, dấu câu bất thường.
  - Rủi ro dương tính giả trong hệ thống lọc thư rác.
- 2.3. Mô hình Transformer cho xử lý ngôn ngữ tự nhiên
  - Biểu diễn ngữ cảnh.
  - Tách token theo đơn vị từ con (subword).
  - Lợi thế của mô hình mã hóa trong bài toán phân loại.
- 2.4. DeBERTa-v3-base
  - Mô hình nền sử dụng: `microsoft/deberta-v3-base`.
  - Đây là mô hình ngôn ngữ tiền huấn luyện của Microsoft; đề tài sử dụng theo hướng transfer learning và fine-tuning, không huấn luyện DeBERTa từ đầu.
  - Bộ mã hóa gồm 12 tầng, khoảng 184 triệu tham số.
  - Phù hợp với bài toán phân loại chuỗi văn bản.
- 2.5. Chỉ số đánh giá
  - Độ chính xác tổng thể, độ chính xác dương tính, độ nhạy và điểm F1.
  - ROC AUC, điểm Brier và mất mát log.
  - Ma trận nhầm lẫn.
  - Tỷ lệ dương tính giả trên tập thư hợp lệ.
- 2.6. Công nghệ triển khai
  - Python, PyTorch, Transformers.
  - Pandas, scikit-learn.
  - Flask cho ứng dụng web/API.
  - Pytest cho kiểm thử tự động.

## Chương 3. Phân Tích Và Thiết Kế Hệ Thống

- 3.1. Yêu cầu chức năng
  - Dự đoán một email.
  - Dự đoán theo lô từ JSON/CSV/TXT.
  - Hiển thị thông tin mô hình và các chỉ số đánh giá.
  - Chọn mẫu từ tập kiểm thử đã xử lý.
  - Chọn mẫu từ dữ liệu thô Kaggle/Enron.
- 3.2. Yêu cầu phi chức năng
  - Kết quả dự đoán ổn định.
  - Môi trường chạy tương thích với các gói phụ thuộc của artifact mô hình.
  - Có kiểm thử hồi quy.
  - Có tài liệu báo cáo và hướng dẫn chạy.
- 3.3. Kiến trúc tổng thể
  - Quy trình dữ liệu: `src/data/`.
  - Tiền xử lý: `src/preprocessing/`.
  - Huấn luyện và suy luận mô hình: `src/models/`.
  - Đánh giá: `src/evaluation/`.
  - Ứng dụng web/API: `src/api/`.
  - Kiểm thử tự động: `tests/`.
- 3.4. Luồng xử lý dự đoán
  - Nhận dữ liệu đầu vào từ biểu mẫu/API.
  - Làm sạch văn bản.
  - Nạp chuỗi xử lý đã huấn luyện.
  - Tính điểm logit, xác suất và nhãn dự đoán.
  - Trả phản hồi hoặc hiển thị bảng điều khiển.
- 3.5. Quản lý artifact mô hình
  - `models/spam_detector.joblib`.
  - `models/spam_detector_metadata.json`.
  - `models/spam_contextual_encoder/`.
  - Ràng buộc môi trường chạy: `transformers==5.0.0`.

## Chương 4. Xây Dựng Dữ Liệu Và Tiền Xử Lý

- 4.1. Nguồn dữ liệu
  - Kaggle `venky73/spam-mails-dataset`: 5,171 dòng dữ liệu thô.
  - Enron spam corpus: 33,716 dòng dữ liệu thô.
  - SpamAssassin ham: `easy_ham`, `easy_ham_2`, `hard_ham`.
- 4.2. Thống kê dữ liệu sau xử lý
  - Tổng mẫu: 39,144.
  - Thư rác: 16,010.
  - Thư hợp lệ: 23,134.
  - Tập huấn luyện: 27,400.
  - Tập thẩm định: 5,872.
  - Tập kiểm thử: 5,872.
- 4.3. Chuẩn hóa cấu trúc dữ liệu
  - Các cột chính: `email_id`, `source`, `label`, `label_name`, `text`, `text_length`.
  - Gộp tiêu đề và nội dung thư khi nguồn dữ liệu có trường tiêu đề.
- 4.4. Tiền xử lý văn bản
  - Loại HTML/script/style.
  - Thay URL bằng `URLTOKEN`.
  - Thay địa chỉ thư điện tử bằng `EMAILTOKEN`.
  - Chuẩn hóa khoảng trắng.
- 4.5. Chia dữ liệu
  - Chia phân tầng theo tỷ lệ 70/15/15.
  - Giữ tỷ lệ nhãn ổn định giữa tập huấn luyện, tập thẩm định và tập kiểm thử.
- 4.6. Ý nghĩa của việc bổ sung mẫu thư hợp lệ từ SpamAssassin
  - Giảm nguy cơ đánh dấu nhầm thư hợp lệ.
  - Bổ sung bản tin, danh sách thư, mẫu thư hợp lệ khó và email kỹ thuật.

## Chương 5. Huấn Luyện Mô Hình

- 5.1. Mô hình sử dụng
  - Chuỗi xử lý: `ContextualTransformerSpamPipeline`.
  - Mô hình nền: `microsoft/deberta-v3-base`.
  - Đầu phân loại: 2 nhãn.
- 5.2. Cấu hình huấn luyện
  - Độ dài tối đa: 256 token.
  - Kích thước lô huấn luyện: 8.
  - Kích thước lô đánh giá: 16.
  - Số epoch: 3.
  - Tốc độ học: `1e-5`.
  - Hệ số suy giảm trọng số: `0.01`.
  - Tích lũy gradient: 2 bước.
  - Tỷ lệ warmup của lịch học: 0.10.
  - Hàm mất mát cross entropy có trọng số theo lớp.
- 5.3. Quy trình huấn luyện
  - Chuẩn bị dữ liệu.
  - Khởi tạo mô hình DeBERTa cho phân loại.
  - Tinh chỉnh toàn bộ bộ mã hóa.
  - Theo dõi điểm F1 trên tập thẩm định.
  - Lưu checkpoint tốt nhất.
- 5.4. Kết quả thẩm định theo epoch
  - Epoch 1: F1 trên tập thẩm định = 0.9874.
  - Epoch 2: F1 trên tập thẩm định = 0.9911, là checkpoint tốt nhất.
  - Epoch 3: F1 trên tập thẩm định = 0.9904.
- 5.5. Ghi chú về Colab và artifact mô hình
  - Notebook: `notebooks/train_on_google_colab.ipynb`.
  - Artifact mô hình được xuất về `models/` cùng siêu dữ liệu.

## Chương 6. Đánh Giá Thực Nghiệm

- 6.1. Kết quả trên tập kiểm thử
  - Độ chính xác tổng thể (accuracy): 0.9932.
  - Độ chính xác dương tính (precision): 0.9892.
  - Độ nhạy (recall): 0.9942.
  - Điểm F1 (F1-score): 0.9917.
  - ROC AUC: 0.9996.
  - Điểm Brier: 0.0063.
  - Mất mát log: 0.0320.
- 6.2. Ma trận nhầm lẫn
  - Thư hợp lệ dự đoán đúng: 3,444.
  - Thư hợp lệ bị nhầm thành thư rác: 26.
  - Thư rác bị nhầm thành thư hợp lệ: 14.
  - Thư rác dự đoán đúng: 2,388.
- 6.3. Đánh giá dương tính giả trên thư hợp lệ
  - `easy_ham`: 0.52% FPR.
  - `easy_ham_2`: 0.00% FPR.
  - `hard_ham`: 0.00% FPR.
  - Tổng FPR trên tập thư hợp lệ giữ lại: 0.31%.
- 6.4. Phân tích ngưỡng
  - Ngưỡng mặc định 0.50 phù hợp cho minh họa.
  - Ngưỡng 0.90 có thể giảm FPR nếu ưu tiên tránh dương tính giả.
- 6.5. Sự cố môi trường chạy gây xác suất 50/50 và cách khắc phục
  - Nguyên nhân: artifact mô hình được xuất bằng `transformers 5.0.0` nhưng từng bị nạp bằng `transformers 5.7.0`.
  - Hiện trạng: `requirements.txt` ghim `transformers==5.0.0`.
  - Bộ nạp mô hình kiểm tra phiên bản để báo lỗi sớm nếu môi trường chạy không khớp.

## Chương 7. Xây Dựng Ứng Dụng Web Và API

- 7.1. Bảng điều khiển ứng dụng web
  - Biểu mẫu dự đoán một email.
  - Tải lên CSV/TXT để dự đoán theo lô.
  - Hiển thị thông tin mô hình và các chỉ số đánh giá.
- 7.2. Tuyến web và điểm cuối API
  - `GET /`.
  - `POST /predict`.
  - `POST /batch`.
  - `GET /model-info`.
  - `GET /api/health`.
  - `GET /api/model-info`.
  - `POST /api/predict`.
  - `POST /api/batch-predict`.
- 7.3. Lấy mẫu từ tập dữ liệu đã xử lý
  - `GET /api/dataset-samples`.
  - `GET /api/dataset-sample/<email_id>`.
  - `POST /predict-sample`.
  - Dữ liệu: `data/processed/test.csv`, 5,872 mẫu.
- 7.4. Lấy mẫu từ dữ liệu thô
  - `GET /api/raw-dataset-samples`.
  - `POST /predict-raw-sample`.
  - Dữ liệu thô Kaggle: 5,171 dòng dữ liệu.
  - Dữ liệu thô Enron: 33,716 dòng dữ liệu.
  - Nạp tối đa 1,000 mẫu mỗi nguồn để minh họa.
- 7.5. Giới hạn yêu cầu
  - Kích thước tệp tải lên mặc định: 4 MiB (4,194,304 byte).
  - Số dòng dự đoán theo lô: 5,000 dòng.
  - Độ dài văn bản: 20,000 ký tự/email.

## Chương 8. Kiểm Thử, Thảo Luận Và Kết Luận

- 8.1. Kiểm thử
  - Bộ kiểm thử pytest hiện thu thập 65 ca; lần chạy đầy đủ gần nhất ghi nhận `65 passed`.
  - Lệnh kiểm thử: `.venv\Scripts\python.exe -m pytest -q`.
  - Có `pytest.ini` để tránh thu thập nhầm file trong `backups/`, `.tmp/`, `.venv/`.
- 8.2. Thảo luận kết quả
  - Mô hình đạt hiệu quả cao trên tập kiểm thử.
  - Tỷ lệ dương tính giả trên tập thư hợp lệ giữ lại thấp.
  - Ứng dụng web/API đủ cho minh họa và kiểm thử chức năng.
- 8.3. Hạn chế
  - Dữ liệu chủ yếu tiếng Anh.
  - Chưa có tập đánh giá tiếng Việt riêng.
  - Ứng dụng Flask là bản minh họa, chưa phải triển khai sản phẩm đầy đủ.
  - Artifact mô hình yêu cầu đúng phiên bản gói phụ thuộc.
- 8.4. Hướng phát triển
  - Bổ sung bộ dữ liệu tiếng Việt.
  - Thêm giám sát và ghi log sản phẩm.
  - Tối ưu tốc độ suy luận.
  - Nghiên cứu khả năng giải thích cho mô hình Transformer.
- 8.5. Kết luận
  - Hệ thống đã hoàn thành quy trình dữ liệu, huấn luyện, đánh giá, ứng dụng web/API và kiểm thử.
  - Kết quả thực nghiệm cho thấy DeBERTa-v3-base phù hợp với bài toán phân loại thư rác email trong phạm vi dữ liệu hiện có.

## Tài Liệu Tham Khảo Đề Xuất

- Tài liệu Hugging Face Transformers.
- Tài liệu PyTorch.
- Tài liệu scikit-learn về các chỉ số phân loại.
- Tài liệu Flask.
- Nguồn bộ dữ liệu Kaggle spam/ham.
- Nguồn bộ dữ liệu Enron spam corpus.
- SpamAssassin Public Corpus.

## Phụ Lục

- Phụ lục A. `REPORT.md`: báo cáo kỹ thuật đầy đủ.
- Phụ lục B. `docs/TRAINING_RESULTS.md`: kết quả huấn luyện và kiểm thử.
- Phụ lục C. `docs/NON_SPAM_EVALUATION.md`: đánh giá dương tính giả.
- Phụ lục D. `docs/WEBAPP_TESTING.md`: kiểm thử ứng dụng web/API.
- Phụ lục E. `docs/DATASET_SAMPLING_FEATURE.md`: lấy mẫu từ tập dữ liệu đã xử lý.
- Phụ lục F. `docs/RAW_DATASET_SAMPLING.md`: lấy mẫu từ dữ liệu thô.
- Phụ lục G. `SETUP.md`: hướng dẫn cài đặt và chạy.
