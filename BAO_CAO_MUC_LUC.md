# MỤC LỤC BÁO CÁO TIỂU LUẬN BÀI TẬP LỚN
## Phân Loại Spam Email Sử Dụng Machine Learning

---

## 📋 CẤU TRÚC BÁO CÁO HOÀN CHỈNH

### TRANG BÌA
- Tên trường, khoa
- Tên môn học: Trí Tuệ Nhân Tạo
- Đề tài: Phân Loại Spam Email Sử Dụng Machine Learning
- Giảng viên hướng dẫn
- Sinh viên thực hiện (Họ tên, MSSV)
- Lớp học phần
- Thời gian thực hiện

### LỀ CẢM ƠN (Optional)
- Cảm ơn giảng viên hướng dẫn
- Cảm ơn các nguồn tài liệu, dataset

### TÓM TẮT (ABSTRACT)
- Tóm tắt bài toán (100-150 từ)
- Phương pháp tiếp cận
- Kết quả chính đạt được
- Kết luận ngắn gọn
- Từ khóa: Spam Detection, Machine Learning, NLP, Text Classification

---

## MỤC LỤC CHI TIẾT

### CHƯƠNG 1: GIỚI THIỆU

#### 1.1. Đặt Vấn Đề
- Tình trạng spam email hiện nay
- Tác hại của spam email
- Nhu cầu phân loại tự động
- Ý nghĩa thực tiễn của đề tài

#### 1.2. Mục Tiêu Nghiên Cứu
- Mục tiêu tổng quát
- Mục tiêu cụ thể:
  - Xây dựng hệ thống phân loại spam/ham
  - Đạt độ chính xác > 95%
  - So sánh hiệu quả các thuật toán
  - Triển khai ứng dụng demo

#### 1.3. Phạm Vi Nghiên Cứu
- Giới hạn về dữ liệu (ngôn ngữ, số lượng)
- Giới hạn về thuật toán
- Giới hạn về thời gian và nguồn lực

#### 1.4. Phương Pháp Nghiên Cứu
- Nghiên cứu lý thuyết
- Thực nghiệm và đánh giá
- So sánh các phương pháp

#### 1.5. Cấu Trúc Báo Cáo
- Tóm tắt nội dung các chương

---

### CHƯƠNG 2: CÁC NGHIÊN CỨU LIÊN QUAN & CỞ SỞ LÝ THUYẾT

#### 2.1. Tổng Quan Về Spam Email
- Định nghĩa spam email
- Lịch sử và sự phát triển
- Các loại spam phổ biến
- Thống kê về spam toàn cầu

#### 2.2. Các Phương Pháp Phát Hiện Spam Truyền Thống
- Blacklist/Whitelist
- Rule-based filtering
- Heuristic methods
- Ưu nhược điểm

#### 2.3. Machine Learning Trong Phân Loại Text
- Giới thiệu Machine Learning
- Text Classification là gì
- Quy trình ML cho text
- Ứng dụng trong spam detection

#### 2.4. Các Nghiên Cứu Trước Đây
- Tổng hợp các paper quan trọng
- Kết quả đạt được của các nghiên cứu
- Phương pháp được sử dụng
- Bảng so sánh các nghiên cứu

#### 2.5. Các Thuật Toán Machine Learning

##### 2.5.1. Naive Bayes
- Nguyên lý hoạt động
- Công thức toán học
- Ưu điểm và nhược điểm
- Ứng dụng trong spam detection

##### 2.5.2. Support Vector Machine (SVM)
- Nguyên lý hoạt động
- Kernel functions
- Ưu điểm và nhược điểm

##### 2.5.3. Decision Tree & Random Forest
- Cấu trúc cây quyết định
- Random Forest ensemble
- Ưu điểm và nhược điểm

##### 2.5.4. Logistic Regression
- Nguyên lý hoạt động
- Hàm sigmoid
- Ưu điểm và nhược điểm

##### 2.5.5. Deep Learning (Optional)
- Neural Networks cơ bản
- LSTM/GRU cho text
- BERT và Transformers

#### 2.6. Kỹ Thuật Xử Lý Ngôn Ngữ Tự Nhiên (NLP)

##### 2.6.1. Text Preprocessing
- Tokenization
- Stopwords removal
- Stemming và Lemmatization
- Normalization

##### 2.6.2. Feature Extraction
- Bag of Words (BoW)
- TF-IDF
- N-grams
- Word Embeddings (Word2Vec, GloVe)

##### 2.6.3. Feature Selection
- Chi-square test
- Mutual Information
- Recursive Feature Elimination

#### 2.7. Các Metrics Đánh Giá
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Khi nào dùng metric nào

---

### CHƯƠNG 3: PHƯƠNG PHÁP THỰC HIỆN

#### 3.1. Tổng Quan Quy Trình
- Sơ đồ quy trình tổng thể
- Các bước thực hiện
- Công cụ và thư viện sử dụng

#### 3.2. Thu Thập Dữ Liệu

##### 3.2.1. Nguồn Dữ Liệu
- SpamAssassin Public Corpus
- Enron Email Dataset
- UCI SMS Spam Collection
- Kaggle datasets
- Mô tả chi tiết từng dataset

##### 3.2.2. Thống Kê Dữ Liệu
- Tổng số emails
- Phân bố spam/ham
- Bảng thống kê chi tiết
- Biểu đồ phân bố

#### 3.3. Khám Phá Dữ Liệu (EDA)

##### 3.3.1. Phân Tích Thống Kê
- Độ dài email trung bình
- Số từ trung bình
- Phân bố ký tự đặc biệt
- Biểu đồ histogram, boxplot

##### 3.3.2. Phân Tích Nội Dung
- Từ khóa phổ biến trong spam
- Từ khóa phổ biến trong ham
- WordCloud visualization
- N-gram analysis

##### 3.3.3. Phát Hiện Patterns
- URLs trong spam
- Số điện thoại
- Ký tự đặc biệt
- Chữ hoa

#### 3.4. Tiền Xử Lý Dữ Liệu

##### 3.4.1. Data Cleaning
- Loại bỏ duplicates
- Xử lý missing values
- Chuẩn hóa encoding
- Loại bỏ HTML tags

##### 3.4.2. Text Preprocessing
- Lowercase conversion
- Tokenization
- Stopwords removal
- Stemming/Lemmatization
- Ví dụ minh họa từng bước

##### 3.4.3. Handling Imbalanced Data
- Phân tích tỷ lệ spam/ham
- Oversampling (SMOTE)
- Undersampling
- Class weights

#### 3.5. Trích Xuất Đặc Trưng (Feature Engineering)

##### 3.5.1. Text Features
- Bag of Words implementation
- TF-IDF implementation
- N-grams (1-3 grams)
- Code và ví dụ

##### 3.5.2. Statistical Features
- Email length
- Word count
- Capital letter ratio
- Special character count
- URL count
- Phone number count

##### 3.5.3. Advanced Features (Optional)
- Word embeddings
- Sentiment scores
- Named entities

##### 3.5.4. Feature Selection
- Correlation analysis
- Chi-square test results
- Top K features selected
- Bảng so sánh features

#### 3.6. Xây Dựng Models

##### 3.6.1. Chia Tách Dữ Liệu
- Train/Validation/Test split (70/15/15)
- Stratified sampling
- Cross-validation strategy (5-fold)

##### 3.6.2. Baseline Models
- Naive Bayes implementation
- Logistic Regression implementation
- Decision Tree implementation
- Hyperparameters cho mỗi model

##### 3.6.3. Advanced Models
- SVM implementation
- Random Forest implementation
- Gradient Boosting (XGBoost)
- Hyperparameters tuning

##### 3.6.4. Deep Learning Models (Optional)
- LSTM architecture
- CNN for text
- BERT fine-tuning
- Training configuration

##### 3.6.5. Hyperparameter Tuning
- Grid Search CV
- Random Search CV
- Bayesian Optimization
- Best parameters found

#### 3.7. Training Process
- Training pipeline
- Early stopping strategy
- Model checkpointing
- Training time cho mỗi model

---

### CHƯƠNG 4: KẾT QUẢ VÀ ĐÁNH GIÁ

#### 4.1. Kết Quả Training

##### 4.1.1. Baseline Models Performance
- Naive Bayes results
- Logistic Regression results
- Decision Tree results
- Bảng so sánh metrics

##### 4.1.2. Advanced Models Performance
- SVM results
- Random Forest results
- XGBoost results
- Deep Learning results (nếu có)

##### 4.1.3. Bảng Tổng Hợp Kết Quả
```
| Model              | Accuracy | Precision | Recall | F1-Score | Training Time |
|--------------------|----------|-----------|--------|----------|---------------|
| Naive Bayes        | 94.2%    | 93.5%     | 92.8%  | 93.1%    | 2s            |
| Logistic Regression| 95.8%    | 95.2%     | 94.9%  | 95.0%    | 5s            |
| SVM                | 96.5%    | 96.1%     | 95.8%  | 95.9%    | 45s           |
| Random Forest      | 97.2%    | 96.8%     | 96.5%  | 96.6%    | 30s           |
| XGBoost            | 97.8%    | 97.3%     | 97.1%  | 97.2%    | 25s           |
```

#### 4.2. Phân Tích Chi Tiết

##### 4.2.1. Confusion Matrix
- Confusion matrix cho từng model
- Phân tích False Positives
- Phân tích False Negatives
- Ví dụ emails bị phân loại sai

##### 4.2.2. ROC Curve & AUC
- ROC curves cho tất cả models
- So sánh AUC scores
- Biểu đồ overlay

##### 4.2.3. Precision-Recall Curve
- PR curves
- Trade-off analysis
- Optimal threshold selection

#### 4.3. Feature Importance Analysis

##### 4.3.1. Top Features
- Top 20 features quan trọng nhất
- Feature importance scores
- Biểu đồ bar chart

##### 4.3.2. Feature Contribution
- SHAP values analysis
- LIME explanations
- Ví dụ cụ thể

#### 4.4. Error Analysis

##### 4.4.1. False Positives Analysis
- Ví dụ ham bị đánh nhầm spam
- Nguyên nhân
- Đề xuất cải thiện

##### 4.4.2. False Negatives Analysis
- Ví dụ spam bị bỏ sót
- Nguyên nhân
- Đề xuất cải thiện

#### 4.5. Model Selection
- Lý do chọn model tốt nhất
- Trade-offs giữa accuracy và speed
- Khuyến nghị deployment

#### 4.6. So Sánh Với Các Nghiên Cứu Trước
- Bảng so sánh với papers
- Phân tích kết quả
- Điểm mạnh/yếu

---

### CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG

#### 5.1. Kiến Trúc Hệ Thống
- Sơ đồ kiến trúc tổng thể
- Các components
- Data flow

#### 5.2. API Development

##### 5.2.1. Backend API
- Framework sử dụng (Flask/FastAPI)
- API endpoints
- Request/Response format
- Code examples

##### 5.2.2. Model Serving
- Model serialization
- Loading model
- Preprocessing pipeline
- Prediction service

#### 5.3. Web Interface (Optional)

##### 5.3.1. Frontend Design
- UI/UX design
- Screenshots
- User flow

##### 5.3.2. Features
- Input email text
- Display prediction
- Show confidence score
- Visualize important features

#### 5.4. Testing

##### 5.4.1. Unit Testing
- Test cases
- Code coverage
- Results

##### 5.4.2. Integration Testing
- API testing
- End-to-end testing
- Performance testing

#### 5.5. Deployment

##### 5.5.1. Deployment Options
- Local deployment
- Cloud deployment (Heroku, AWS, GCP)
- Docker containerization

##### 5.5.2. Monitoring
- Logging
- Performance metrics
- Error tracking

---

### CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

#### 6.1. Tổng Kết

##### 6.1.1. Kết Quả Đạt Được
- Mục tiêu đã hoàn thành
- Metrics đạt được
- Sản phẩm cuối cùng

##### 6.1.2. Đóng Góp Của Đề Tài
- Đóng góp về mặt kỹ thuật
- Đóng góp về mặt thực tiễn
- Bài học kinh nghiệm

#### 6.2. Khó Khăn Và Thách Thức

##### 6.2.1. Khó Khăn Gặp Phải
- Vấn đề về dữ liệu
- Vấn đề về thuật toán
- Vấn đề về tài nguyên

##### 6.2.2. Cách Giải Quyết
- Solutions implemented
- Workarounds
- Lessons learned

#### 6.3. Hạn Chế Của Đề Tài
- Hạn chế về dữ liệu
- Hạn chế về model
- Hạn chế về deployment
- Hạn chế về ngôn ngữ (chỉ tiếng Anh)

#### 6.4. Hướng Phát Triển Trong Tương Lai

##### 6.4.1. Cải Thiện Model
- Thử deep learning models phức tạp hơn
- Ensemble methods
- Transfer learning với BERT

##### 6.4.2. Mở Rộng Tính Năng
- Hỗ trợ đa ngôn ngữ (tiếng Việt)
- Phân loại nhiều categories
- Real-time detection
- Email attachment scanning

##### 6.4.3. Tối Ưu Hệ Thống
- Improve inference speed
- Model compression
- Distributed training
- Auto-retraining pipeline

##### 6.4.4. Ứng Dụng Thực Tế
- Tích hợp với email clients
- Mobile app
- Browser extension
- Enterprise solution

---

### TÀI LIỆU THAM KHẢO

#### Sách và Giáo Trình
1. "Machine Learning" - Tom Mitchell
2. "Pattern Recognition and Machine Learning" - Christopher Bishop
3. "Natural Language Processing with Python" - Steven Bird
4. "Hands-On Machine Learning with Scikit-Learn and TensorFlow" - Aurélien Géron

#### Papers và Nghiên Cứu
1. "A Plan for Spam" - Paul Graham (2002)
2. "Spam Filtering with Naive Bayes" - Metsis et al. (2006)
3. "Email Spam Detection Using Machine Learning Algorithms" - Various authors
4. Các papers từ IEEE, ACM, arXiv

#### Datasets
1. SpamAssassin Public Corpus
2. Enron Email Dataset
3. UCI Machine Learning Repository
4. Kaggle Datasets

#### Thư Viện và Frameworks
1. Scikit-learn Documentation
2. NLTK Documentation
3. TensorFlow/PyTorch Documentation
4. Pandas, NumPy Documentation

#### Websites và Tutorials
1. Kaggle Tutorials
2. Towards Data Science Articles
3. Machine Learning Mastery
4. Stack Overflow

---

### PHỤ LỤC

#### Phụ Lục A: Source Code
- Link GitHub repository
- Cấu trúc thư mục
- Hướng dẫn chạy code

#### Phụ Lục B: Dataset Samples
- Ví dụ spam emails
- Ví dụ ham emails
- Thống kê chi tiết

#### Phụ Lục C: Kết Quả Chi Tiết
- Full confusion matrices
- Detailed metrics cho mỗi fold
- Training logs

#### Phụ Lục D: API Documentation
- API endpoints chi tiết
- Request/Response examples
- Error codes

#### Phụ Lục E: Screenshots
- Ảnh chụp màn hình ứng dụng
- Biểu đồ và visualizations
- Demo workflow

#### Phụ Lục F: Glossary
- Thuật ngữ tiếng Anh - Tiếng Việt
- Định nghĩa các khái niệm

---

## 📝 YÊU CẦU TRÌNH BÀY

### Format
- Font chữ: Times New Roman, size 13 (nội dung), size 14-16 (tiêu đề)
- Lề: Trái 3cm, Phải 2cm, Trên 2cm, Dưới 2cm
- Giãn dòng: 1.5 lines
- Căn đều 2 bên
- Đánh số trang

### Hình Ảnh và Bảng Biểu
- Đánh số và có caption
- Rõ ràng, dễ đọc
- Trích dẫn nguồn nếu cần

### Code
- Sử dụng font monospace (Consolas, Courier New)
- Syntax highlighting
- Comment đầy đủ

### Trích Dẫn
- Sử dụng IEEE hoặc APA style
- Đầy đủ và chính xác
- Tránh plagiarism

---

## 📊 CHECKLIST HOÀN THÀNH

### Nội Dung
- [ ] Tất cả các chương đã hoàn thành
- [ ] Có đủ hình ảnh, biểu đồ minh họa
- [ ] Code đã được test và chạy tốt
- [ ] Kết quả thực nghiệm đầy đủ
- [ ] Phân tích chi tiết và sâu sắc

### Trình Bày
- [ ] Format đúng quy định
- [ ] Không có lỗi chính tả
- [ ] Đánh số trang, hình, bảng đúng
- [ ] Mục lục tự động cập nhật
- [ ] Tài liệu tham khảo đầy đủ

### Kỹ Thuật
- [ ] Code chạy được
- [ ] Có README hướng dẫn
- [ ] Dataset có sẵn hoặc hướng dẫn tải
- [ ] Model đã được lưu
- [ ] Demo/API hoạt động

### Nộp Bài
- [ ] File PDF báo cáo
- [ ] Source code (ZIP hoặc GitHub link)
- [ ] Slides thuyết trình
- [ ] Demo video (nếu có)

---

**Lưu ý:** Đây là mục lục chi tiết và đầy đủ. Tùy theo yêu cầu của giảng viên và thời gian, bạn có thể điều chỉnh độ sâu của từng phần cho phù hợp.

**Thời lượng báo cáo:** 40-60 trang (không tính phụ lục)

**Good luck! 📚🚀**
