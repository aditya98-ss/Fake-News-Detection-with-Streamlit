# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake['label'] = 0  # Fake
    true['label'] = 1  # Real
    data = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data['text']
    y = data['label']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = tfidf_vectorizer.fit_transform(X)
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_vec, y)
    return model, tfidf_vectorizer

# Load and train
data = load_data()
model, vectorizer = train_model(data)

# Streamlit App UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Paste any news article below and see if it's **Real** or **Fake**.")

# Input
user_input = st.text_area("📝 Paste News Article Text Here", height=300)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please paste some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader("🧠 Prediction:")
        st.success(label)
