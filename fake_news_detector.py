# fake_news_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the data
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# 2. Add labels
true['label'] = 1  # Real
fake['label'] = 0  # Fake

# 3. Combine and shuffle the dataset
data = pd.concat([true, fake], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Split features and labels
X = data['text']
y = data['label']

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 6. Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# 7. Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# 8. Predict and evaluate
y_pred = model.predict(tfidf_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(acc * 100, 2), "%")
print("Confusion Matrix:\n", cm)
