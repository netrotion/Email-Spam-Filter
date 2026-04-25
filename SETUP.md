# Email Spam Filter - Setup Guide

## Cấu Trúc Dự Án

```
BTL-TTNT/
├── data/
│   ├── raw/              # Dữ liệu gốc
│   └── processed/        # Dữ liệu đã xử lý
├── models/               # Models đã train
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── preprocessing/    # Code tiền xử lý
│   ├── models/          # Code training models
│   ├── evaluation/      # Code đánh giá
│   └── api/             # API code
├── tests/               # Unit tests
├── .gitignore
├── requirements.txt
├── README.md
└── BAO_CAO_MUC_LUC.md
```

## Hướng Dẫn Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/netrotion/Email-Spam-Filter.git
cd Email-Spam-Filter
```

### 2. Tạo Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 5. Download SpaCy Model (Optional)
```bash
python -m spacy download en_core_web_sm
```

## Sử Dụng

### Training Model
```bash
python src/models/train.py
```

### Evaluation
```bash
python src/evaluation/evaluate.py
```

### Run API
```bash
python src/api/app.py
```

## Tài Liệu

- [README.md](README.md) - Roadmap chi tiết
- [BAO_CAO_MUC_LUC.md](BAO_CAO_MUC_LUC.md) - Mục lục báo cáo

## Contributors

- Lê Việt Hùng

## License

MIT License
