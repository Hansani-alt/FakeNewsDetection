import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

# Convert tokens to normal text
data["processed_text"] = data["tokens"].apply(
    lambda x: " ".join(eval(x)) if isinstance(x, str) else " ".join(x)
)

# Train TF-IDF again
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(data["processed_text"])

# Load saved model
model = joblib.load("saved_models/random_forest_model.pkl")

# User input
news = input("Enter News: ")

# Transform input
news_vector = tfidf.transform([news])

# Predict
prediction = model.predict(news_vector)

print("\nPrediction:")

if prediction[0] == 0:
    print("REAL NEWS")
else:
    print("FAKE NEWS")