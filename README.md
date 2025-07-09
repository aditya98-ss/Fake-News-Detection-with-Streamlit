# 📰 Fake News Detector with Streamlit

A simple machine learning project that classifies news articles as **Real** or **Fake** using NLP and Scikit-learn.

## 🚀 Features
- Paste any news text and get a prediction
- Trained on real-world Fake and True news data
- Built with Streamlit for web interface

## 📦 Dataset
- Source: [Kaggle Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Files: `Fake.csv`, `True.csv`

## 🔧 Tech Stack
- Python
- Streamlit
- Scikit-learn
- TfidfVectorizer
- PassiveAggressiveClassifier

## ▶️ Run Locally
```bash
git clone https://github.com/adityaparichha/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
