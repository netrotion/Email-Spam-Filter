# Bài Tập Lớn: Phân Loại Spam Email

## 📖 Giải Thích Khái Niệm Cơ Bản (Dành Cho Người Mới)

### 🤔 Machine Learning Là Gì?
**Machine Learning (Học máy)** là việc dạy máy tính học từ dữ liệu, giống như con người học từ kinh nghiệm. Thay vì lập trình từng quy tắc cụ thể, ta cho máy xem nhiều ví dụ và để nó tự tìm ra quy luật.

**Ví dụ thực tế:** 
- Thay vì viết code "nếu email có từ 'free money' thì là spam", ta cho máy xem 10,000 emails (có gắn nhãn spam/không spam), máy sẽ tự học được những đặc điểm nào thường xuất hiện trong spam.

---

### 🎯 Các Khái Niệm Quan Trọng

#### 1. Dataset (Tập Dữ Liệu)
**Là gì:** Tập hợp các ví dụ dùng để dạy máy học.
**Trong dự án này:** Hàng nghìn emails đã được gắn nhãn "spam" hoặc "ham" (không spam).
**Ví dụ:**
```
Email 1: "Congratulations! You won $1,000,000!" → Spam
Email 2: "Meeting at 3pm tomorrow" → Ham
Email 3: "Click here for free iPhone!" → Spam
```

#### 2. Features (Đặc Trưng)
**Là gì:** Các thông tin đặc biệt mà máy sẽ xem xét để đưa ra quyết định.
**Ví dụ features trong email:**
- Số lần xuất hiện từ "free" (miễn phí)
- Có chứa link hay không
- Độ dài email
- Số lượng dấu chấm than (!!!)
- Có số điện thoại hay không

**Tại sao quan trọng:** Giống như bác sĩ xem triệu chứng (sốt, ho, đau đầu) để chẩn đoán bệnh, máy xem features để phân loại email.

#### 3. Model (Mô Hình)
**Là gì:** "Bộ não" của chương trình, là kết quả sau khi máy học xong.
**Ví dụ đơn giản:** Sau khi học, model có thể biết:
- Email có từ "free" + link + nhiều dấu ! → 95% là spam
- Email ngắn + không có link → 90% là ham

#### 4. Training (Huấn Luyện)
**Là gì:** Quá trình cho máy học từ dữ liệu.
**Cách hoạt động:**
1. Cho máy xem email + nhãn đúng
2. Máy đoán → sai → điều chỉnh
3. Lặp lại hàng nghìn lần
4. Máy ngày càng đoán chính xác hơn

**Ví dụ:** Giống như học sinh làm bài tập có đáp án, làm sai thì sửa, làm nhiều thì giỏi.

#### 5. Testing (Kiểm Tra)
**Là gì:** Kiểm tra xem model có hoạt động tốt với dữ liệu mới chưa từng thấy không.
**Tại sao cần:** Đảm bảo model không chỉ "thuộc lòng" mà thực sự hiểu quy luật.

---

### 📊 Các Chỉ Số Đánh Giá (Metrics)

#### Accuracy (Độ Chính Xác)
**Là gì:** Tỷ lệ dự đoán đúng trên tổng số dự đoán.
**Công thức:** (Số email đoán đúng / Tổng số email) × 100%
**Ví dụ:** Đoán đúng 95/100 email → Accuracy = 95%

#### Precision (Độ Chính Xác Dương)
**Là gì:** Trong số email máy nói là spam, có bao nhiêu thực sự là spam?
**Tại sao quan trọng:** Tránh đánh nhầm email quan trọng thành spam (false positive).
**Ví dụ:** Máy nói 100 email là spam, nhưng chỉ 90 email thực sự spam → Precision = 90%

#### Recall (Độ Phủ)
**Là gì:** Trong tất cả email spam thực sự, máy bắt được bao nhiêu?
**Tại sao quan trọng:** Đảm bảo không bỏ sót spam.
**Ví dụ:** Có 100 spam thực sự, máy chỉ bắt được 85 → Recall = 85%

#### F1-Score
**Là gì:** Điểm cân bằng giữa Precision và Recall.
**Khi nào dùng:** Khi cần cân bằng giữa "không bỏ sót spam" và "không đánh nhầm email tốt".

#### Confusion Matrix (Ma Trận Nhầm Lẫn)
**Là gì:** Bảng thống kê chi tiết các trường hợp đúng/sai.
```
                Dự đoán Spam    Dự đoán Ham
Thực tế Spam         85              15        ← Bỏ sót 15 spam
Thực tế Ham          10              90        ← Đánh nhầm 10 email tốt
```

---

### 🔧 Các Kỹ Thuật Xử Lý Dữ Liệu

#### Tokenization (Tách Từ)
**Là gì:** Chia câu thành từng từ riêng biệt.
**Ví dụ:**
- Input: "Hello, how are you?"
- Output: ["Hello", "how", "are", "you"]

#### Stopwords (Từ Dừng)
**Là gì:** Những từ xuất hiện nhiều nhưng không mang ý nghĩa quan trọng.
**Ví dụ:** "the", "is", "are", "a", "an"
**Tại sao loại bỏ:** Giúp máy tập trung vào từ quan trọng như "free", "winner", "urgent".

#### Stemming/Lemmatization (Chuẩn Hóa Từ)
**Là gì:** Đưa các dạng khác nhau của từ về dạng gốc.
**Ví dụ:**
- "running", "runs", "ran" → "run"
- "better", "good" → "good"
**Tại sao cần:** Máy hiểu "win", "winning", "winner" là cùng một ý nghĩa.

#### TF-IDF (Term Frequency - Inverse Document Frequency)
**Là gì:** Đo lường mức độ quan trọng của một từ trong email.
**Cách hoạt động:**
- Từ xuất hiện nhiều trong 1 email nhưng ít trong các email khác → quan trọng
- Từ xuất hiện ở mọi email → không quan trọng

**Ví dụ:**
- Từ "meeting" xuất hiện 5 lần trong email này, nhưng chỉ 10% email có từ này → TF-IDF cao
- Từ "the" xuất hiện ở 99% email → TF-IDF thấp

#### N-grams
**Là gì:** Nhóm N từ liên tiếp.
**Ví dụ:**
- Unigram (1 từ): ["free", "money", "now"]
- Bigram (2 từ): ["free money", "money now"]
- Trigram (3 từ): ["free money now"]
**Tại sao cần:** "Free money" có ý nghĩa khác với "free" và "money" riêng lẻ.

---

### 🤖 Các Thuật Toán Machine Learning

#### Naive Bayes (Bayes Ngây Thơ)
**Là gì:** Thuật toán dựa trên xác suất.
**Cách hoạt động:** Tính xác suất email là spam dựa trên các từ xuất hiện.
**Ví dụ đơn giản:**
- P(spam | có từ "free") = 80%
- P(spam | có từ "click") = 70%
- P(spam | có cả "free" và "click") = 90%
**Ưu điểm:** Nhanh, đơn giản, hiệu quả với text.
**Nhược điểm:** Giả định các từ độc lập (không phải lúc nào cũng đúng).

#### Logistic Regression (Hồi Quy Logistic)
**Là gì:** Tính điểm số từ 0-1, càng gần 1 càng có khả năng là spam.
**Ví dụ:**
- Email có 3 từ "free", 2 links → Điểm = 0.92 → Spam
- Email bình thường → Điểm = 0.15 → Ham

#### Decision Tree (Cây Quyết Định)
**Là gì:** Chuỗi câu hỏi yes/no để phân loại.
**Ví dụ:**
```
Có từ "free"?
├─ Yes → Có link?
│   ├─ Yes → SPAM (95%)
│   └─ No → Có số điện thoại?
│       ├─ Yes → SPAM (80%)
│       └─ No → HAM (60%)
└─ No → HAM (90%)
```

#### Random Forest (Rừng Ngẫu Nhiên)
**Là gì:** Nhiều Decision Trees bỏ phiếu cùng nhau.
**Ví dụ:** 100 cây quyết định, 85 cây nói spam → Kết quả: Spam
**Ưu điểm:** Chính xác hơn 1 cây đơn lẻ.

#### SVM (Support Vector Machine)
**Là gì:** Tìm đường phân chia tốt nhất giữa spam và ham.
**Ví dụ hình ảnh:** Vẽ 1 đường thẳng tách 2 nhóm điểm (spam bên này, ham bên kia).

#### Deep Learning (Học Sâu)
**Là gì:** Mạng neural nhiều lớp, bắt chước não người.
**Khi nào dùng:** Dữ liệu lớn, bài toán phức tạp.
**Ví dụ:** LSTM hiểu được ngữ cảnh câu, BERT hiểu ý nghĩa sâu của văn bản.

---

### 🔄 Quy Trình Machine Learning Cơ Bản

```
1. Thu thập dữ liệu
   ↓
2. Làm sạch dữ liệu (loại lỗi, trùng lặp)
   ↓
3. Trích xuất features (chọn thông tin quan trọng)
   ↓
4. Chia dữ liệu: 80% training, 20% testing
   ↓
5. Chọn thuật toán (Naive Bayes, SVM, etc.)
   ↓
6. Training (cho máy học)
   ↓
7. Testing (kiểm tra độ chính xác)
   ↓
8. Đánh giá (Accuracy, Precision, Recall)
   ↓
9. Cải thiện (thử thuật toán khác, thêm features)
   ↓
10. Deploy (đưa vào sử dụng thực tế)
```

---

### 🎓 Các Vấn Đề Thường Gặp

#### Overfitting (Học Vẹt)
**Là gì:** Model học thuộc lòng dữ liệu training, không hiểu quy luật chung.
**Ví dụ:** Học sinh thuộc đáp án 100 câu, gặp câu mới thì không biết làm.
**Giải pháp:** Dùng nhiều dữ liệu, regularization, cross-validation.

#### Underfitting (Học Kém)
**Là gì:** Model quá đơn giản, không học được gì.
**Ví dụ:** Chỉ nhìn 1 từ "free" để quyết định, bỏ qua các thông tin khác.
**Giải pháp:** Dùng model phức tạp hơn, thêm features.

#### Class Imbalance (Mất Cân Bằng Lớp)
**Là gì:** Số lượng spam và ham chênh lệch quá lớn.
**Ví dụ:** 9,000 ham vs 1,000 spam → Model có thể "lười" đoán tất cả là ham.
**Giải pháp:** Oversampling (tăng spam), undersampling (giảm ham), hoặc dùng weights.

#### Data Leakage (Rò Rỉ Dữ Liệu)
**Là gì:** Thông tin từ test set "lọt" vào training.
**Ví dụ:** Dùng toàn bộ data để tính TF-IDF trước khi chia train/test.
**Hậu quả:** Accuracy cao giả tạo, thực tế kém.
**Giải pháp:** Chia train/test trước, chỉ dùng train data để fit.

---

### 💡 Tips Cho Người Mới Bắt Đầu

1. **Bắt đầu đơn giản:** Thử Naive Bayes trước, sau đó mới thử các model phức tạp.
2. **Hiểu dữ liệu:** Dành thời gian khám phá dữ liệu, xem ví dụ spam/ham thực tế.
3. **Visualize:** Vẽ biểu đồ để thấy patterns (wordcloud, distribution).
4. **Thử nghiệm:** Thử nhiều features, nhiều thuật toán, so sánh kết quả.
5. **Đọc code mẫu:** Tìm tutorials trên Kaggle, GitHub để học.
6. **Hỏi khi không hiểu:** Google, Stack Overflow, ChatGPT là bạn đồng hành.

---

## Roadmap Thực Hiện Dự Án (A → Z)

### 🎯 Giai Đoạn 1: Khởi Động Dự Án (Week 1)

#### 1.1 Định Nghĩa Bài Toán
**Vai trò:** Project Manager + Data Scientist
- Xác định mục tiêu: Phân loại email thành spam/ham với độ chính xác > 95%
- Định nghĩa metrics đánh giá: Accuracy, Precision, Recall, F1-Score
- Xác định scope: Xử lý email tiếng Anh/tiếng Việt
- Lập timeline và phân công công việc

#### 1.2 Nghiên Cứu Tài Liệu
**Vai trò:** Research Team
- Tìm hiểu các thuật toán phân loại: Naive Bayes, SVM, Random Forest, Deep Learning
- Nghiên cứu các kỹ thuật xử lý text: TF-IDF, Word2Vec, BERT
- Đọc papers và case studies về spam detection
- Tổng hợp best practices

#### 1.3 Setup Môi Trường
**Vai trò:** DevOps + Developer
- Cài đặt Python 3.8+, Jupyter Notebook
- Setup virtual environment (venv/conda)
- Cài đặt thư viện: pandas, numpy, scikit-learn, nltk, tensorflow/pytorch
- Setup Git repository và version control
- Tạo cấu trúc thư mục dự án

---

### 📊 Giai Đoạn 2: Thu Thập & Chuẩn Bị Dữ Liệu (Week 2-3)

#### 2.1 Thu Thập Dataset
**Vai trò:** Data Engineer
- Tải các dataset công khai:
  - SpamAssassin Public Corpus
  - Enron Email Dataset
  - UCI SMS Spam Collection
  - Kaggle Email Spam Dataset
- Tổng hợp dataset riêng (nếu có)
- Đảm bảo có ít nhất 5,000-10,000 emails
- Kiểm tra tính hợp pháp và quyền sử dụng dữ liệu

#### 2.2 Khám Phá Dữ Liệu (EDA)
**Vai trò:** Data Analyst
- Phân tích phân bố spam/ham (class imbalance?)
- Thống kê độ dài email, số từ, ký tự đặc biệt
- Tìm patterns phổ biến trong spam: từ khóa, URLs, số điện thoại
- Visualize dữ liệu: wordcloud, distribution plots
- Phát hiện outliers và missing values

#### 2.3 Làm Sạch Dữ Liệu
**Vai trò:** Data Engineer
- Loại bỏ duplicates
- Xử lý missing values
- Chuẩn hóa encoding (UTF-8)
- Loại bỏ HTML tags, URLs (hoặc extract làm features)
- Xử lý email headers và metadata
- Cân bằng dataset nếu cần (oversampling/undersampling)

#### 2.4 Tiền Xử Lý Text
**Vai trò:** NLP Engineer
- Lowercase conversion
- Tokenization (tách từ)
- Remove stopwords (từ dừng)
- Stemming/Lemmatization (chuẩn hóa từ)
- Remove punctuation và special characters
- Xử lý numbers và dates
- Lưu processed data

---

### 🔧 Giai Đoạn 3: Feature Engineering (Week 3-4)

#### 3.1 Trích Xuất Features Cơ Bản
**Vai trò:** ML Engineer
- **Text Features:**
  - Bag of Words (BoW)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - N-grams (unigram, bigram, trigram)
- **Statistical Features:**
  - Độ dài email (số ký tự, số từ)
  - Tỷ lệ chữ hoa
  - Số lượng ký tự đặc biệt (!@#$%)
  - Số lượng URLs
  - Số lượng số điện thoại

#### 3.2 Features Nâng Cao
**Vai trò:** NLP Engineer
- Word embeddings: Word2Vec, GloVe
- Sentiment analysis scores
- Named Entity Recognition (NER)
- Email header features (sender domain, reply-to)
- Spam keywords frequency
- HTML/Plain text ratio

#### 3.3 Feature Selection
**Vai trò:** Data Scientist
- Phân tích correlation matrix
- Chi-square test cho categorical features
- Mutual Information scores
- Recursive Feature Elimination (RFE)
- Chọn top K features quan trọng nhất

---

### 🤖 Giai Đoạn 4: Xây Dựng & Huấn Luyện Model (Week 4-6)

#### 4.1 Chia Tách Dữ Liệu
**Vai trò:** ML Engineer
- Train/Validation/Test split (70/15/15 hoặc 80/10/10)
- Stratified split để giữ tỷ lệ spam/ham
- Cross-validation setup (K-Fold)
- Lưu các splits để reproducibility

#### 4.2 Baseline Models
**Vai trò:** ML Engineer
- **Naive Bayes:**
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
  - Gaussian Naive Bayes
- **Logistic Regression**
- **Decision Tree**
- Train và đánh giá baseline performance

#### 4.3 Advanced Models
**Vai trò:** ML Engineer + Data Scientist
- **Traditional ML:**
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
  - Ensemble methods
- **Deep Learning:**
  - LSTM/GRU networks
  - CNN for text classification
  - BERT/Transformer models (nếu có resources)

#### 4.4 Hyperparameter Tuning
**Vai trò:** ML Engineer
- Grid Search CV
- Random Search CV
- Bayesian Optimization
- Tune các parameters:
  - Learning rate
  - Regularization (C, alpha)
  - Tree depth, n_estimators
  - Batch size, epochs (DL)

#### 4.5 Training Process
**Vai trò:** ML Engineer
- Setup training pipeline
- Implement early stopping
- Monitor training/validation loss
- Save checkpoints
- Log experiments (MLflow, Weights & Biases)
- Handle overfitting: dropout, regularization

---

### 📈 Giai Đoạn 5: Đánh Giá & Tối Ưu (Week 6-7)

#### 5.1 Model Evaluation
**Vai trò:** Data Scientist
- **Metrics:**
  - Accuracy
  - Precision (quan trọng: tránh false positive)
  - Recall (quan trọng: bắt hết spam)
  - F1-Score
  - ROC-AUC curve
  - Confusion Matrix
- So sánh performance các models
- Phân tích errors: false positives/negatives

#### 5.2 Model Interpretation
**Vai trò:** Data Scientist
- Feature importance analysis
- SHAP values
- LIME explanations
- Phân tích từ khóa quan trọng nhất
- Hiểu tại sao model đưa ra quyết định

#### 5.3 Model Optimization
**Vai trò:** ML Engineer
- Ensemble các models tốt nhất
- Stacking/Blending techniques
- Threshold tuning cho classification
- Optimize inference speed
- Model compression (nếu cần deploy)

---

### 🚀 Giai Đoạn 6: Deployment & Testing (Week 7-8)

#### 6.1 Model Serialization
**Vai trò:** ML Engineer
- Save model: pickle, joblib, ONNX
- Save preprocessing pipeline
- Version control models
- Document model artifacts

#### 6.2 API Development
**Vai trò:** Backend Developer
- Xây dựng REST API (Flask/FastAPI)
- Endpoints:
  - POST /predict (classify single email)
  - POST /batch_predict (classify multiple)
  - GET /model_info
- Input validation
- Error handling

#### 6.3 Web Interface (Optional)
**Vai trò:** Frontend Developer
- Simple UI để test model
- Upload email hoặc paste text
- Hiển thị kết quả: Spam/Ham + confidence score
- Visualize features quan trọng

#### 6.4 Testing
**Vai trò:** QA Engineer
- Unit tests cho preprocessing functions
- Integration tests cho API
- Test với real-world emails
- Performance testing (latency, throughput)
- Edge cases testing

---

### 📝 Giai Đoạn 7: Documentation & Presentation (Week 8)

#### 7.1 Technical Documentation
**Vai trò:** Technical Writer + Team
- README.md với hướng dẫn setup
- Code documentation (docstrings)
- API documentation (Swagger/OpenAPI)
- Model card: architecture, performance, limitations
- Data pipeline documentation

#### 7.2 Report & Analysis
**Vai trò:** Data Scientist + Team
- **Báo cáo bao gồm:**
  - Giới thiệu bài toán
  - Mô tả dataset và EDA
  - Feature engineering approach
  - Model selection và comparison
  - Results và evaluation metrics
  - Challenges và solutions
  - Future improvements
  - References

#### 7.3 Presentation
**Vai trò:** Toàn Team
- Chuẩn bị slides (15-20 phút)
- Demo live system
- Highlight key findings
- Q&A preparation

---

## 🛠️ Tech Stack Đề Xuất

### Core Libraries
- **Data Processing:** pandas, numpy
- **NLP:** nltk, spaCy, gensim
- **ML:** scikit-learn, xgboost, lightgbm
- **DL:** tensorflow/keras hoặc pytorch
- **Visualization:** matplotlib, seaborn, plotly

### Development Tools
- **IDE:** Jupyter Notebook, VS Code, PyCharm
- **Version Control:** Git, GitHub/GitLab
- **Experiment Tracking:** MLflow, Weights & Biases
- **API:** Flask, FastAPI
- **Testing:** pytest, unittest

---

## 📋 Deliverables

1. ✅ Clean, documented code repository
2. ✅ Trained models với performance > 95% accuracy
3. ✅ API service hoặc web interface
4. ✅ Technical report (PDF)
5. ✅ Presentation slides
6. ✅ Demo video (optional)

---

## ⚠️ Lưu Ý Quan Trọng

### Best Practices
- Commit code thường xuyên với clear messages
- Document mọi quyết định quan trọng
- Backup data và models
- Test thoroughly trước khi demo
- Handle edge cases và errors gracefully

### Common Pitfalls
- ❌ Overfitting trên training data
- ❌ Data leakage giữa train/test
- ❌ Không xử lý class imbalance
- ❌ Quên preprocess test data giống train data
- ❌ Không validate với real-world data

### Evaluation Criteria
- Code quality và organization (20%)
- Model performance (30%)
- Feature engineering creativity (20%)
- Documentation và presentation (20%)
- Innovation và insights (10%)

---

## 📚 Tài Liệu Tham Khảo

- [Scikit-learn Text Classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [NLTK Documentation](https://www.nltk.org/)
- [SpamAssassin Dataset](https://spamassassin.apache.org/old/publiccorpus/)
- Papers: "Naive Bayes Spam Filtering", "Deep Learning for Email Classification"

---

**Good luck! 🚀**
