# 🛡️ FactCheck — Misinformation Detector

An end-to-end system that detects fake news and misinformation using **NLP feature extraction** and **Ensemble Machine Learning**. Built to demonstrate real-world ML/NLP engineering skills.

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)

---

## 📌 Project Overview

FactCheck analyzes news articles by extracting linguistic patterns, sentiment signals, and TF-IDF text features. It then runs predictions through an ensemble of 5 trained ML models to classify news as **Fake** or **True**, returning a confidence score.

---

## 🏗️ Architecture
```
Raw News Input (Title + Article)
        │
        ▼
┌─────────────────────┐
│  Text Preprocessing │  → Lowercase, URL removal, stopwords, lemmatization
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Feature Extraction │  → TF-IDF + Sentiment + Linguistic Features
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Ensemble Model     │  → LR + RF + XGBoost + GB + SVM (averaged voting)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Prediction Output  │  → Label + Confidence Score + Feature Breakdown
└─────────────────────┘
```

---

## 🧠 ML & NLP Techniques Used

| Category | Techniques |
|---|---|
| **NLP** | Tokenization, Stopword Removal, Lemmatization, TF-IDF Vectorization |
| **Feature Engineering** | Sentiment Analysis (VADER), Subjectivity (TextBlob), Uppercase Ratio, Lexical Diversity, Punctuation Patterns |
| **ML Models** | Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM |
| **Ensemble** | Soft Voting (averaged predicted probabilities from all models) |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, AUC-ROC |

---

## 📂 Project Structure
```
FactCheck/
│
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Text cleaning & preprocessing
│   ├── feature_extraction.py     # TF-IDF + handcrafted features
│   ├── model.py                  # Model training + ensemble logic
│   └── utils.py                  # Prediction pipeline for the app
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb    # Data cleaning & word clouds
│   ├── 03_Feature_Extraction.ipynb  # Feature engineering
│   └── 04_Model_Training.ipynb   # Model training & evaluation
├── models/                       # Saved trained models (.pkl)
├── data/
│   ├── raw/                      # Original Fake.csv & True.csv
│   └── processed/                # Preprocessed & feature data
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 🚀 Getting Started

### Option A: Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/PiyushCodess/FactCheck.git
cd FactCheck

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/streamlit_app.py
```

### Option B: Run with Docker
```bash
# 1. Build the image
docker build -t FactCheck .

# 2. Run the container
docker run -p 8501:8501 FactCheck
```

Open `http://localhost:8501` in your browser.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9394311210262131 | 0.9207920792079208 | 0.9554050898902638 | 0.9377793056033001 | 0.9780558585580635 |
| Random Forest | 0.9981037367540435 | 0.9969711090400746 | 0.9990660751809479 | 0.9980174927113702 | 0.9998646586950604 |
| XGBoost | 0.9979921918572225 | 0.9974341031024027 | 0.9983656315666589 | 0.997899649941657 | 0.9999558170979452 |
| Gradient Boosting | 0.9974344673731177 | 0.9969668688754083 | 0.9976651879523698 | 0.9973159061734158 | 0.9997717299840233 |
| SVM | 0.82643614054657 | 0.7643979057591623 | 0.9203829091758113 | 0.8351694915254237 | 0.8948021807802782 |
| **Ensemble** | - | - | - | - | - |

> ⚡ Fill in your actual scores from `models/model_results.csv`

---

## 📦 Dataset

**ISOT Fake News Dataset**
- Source: University of Victoria
- `True.csv` — Real news articles collected from Reuters
- `Fake.csv` — Fake news articles collected from various unreliable sources
- Total articles: ~44,000+

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9 |
| NLP | NLTK, spaCy, TextBlob, TF-IDF |
| ML | Scikit-Learn, XGBoost |
| Web App | Streamlit |
| Deployment | Docker |
| Version Control | Git / GitHub |

---

## 🤝 Future Improvements

- Fine-tune a BERT/RoBERTa transformer model for higher accuracy
- Add real-time news scraping via RSS feeds
- Integrate source credibility scoring
- Add multi-language support
- Deploy to cloud (AWS/GCP) with CI/CD pipeline

---

## 📜 License

This project is for **educational purposes only**. Not intended for production misinformation detection.

---

## 👨‍💻 Author

**Piyush Patrikar**
- GitHub: [PiyushCodess](https://github.com/PiyushCodess)
- LinkedIn: [PiyushPatrikar](www.linkedin.com/in/piyush-patrikar)