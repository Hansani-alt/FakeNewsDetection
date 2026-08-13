import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

# Convert tokens back to normal text
data["processed_text"] = data["tokens"].apply(
    lambda x: " ".join(eval(x)) if isinstance(x, str) else " ".join(x)
)

print("\nProcessed Text:")
print(data["processed_text"].head())

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data["processed_text"])

y = data["label"]

print("\nTF-IDF Shape:")
print(X.shape)

print("\nLabels Shape:")
print(y.shape)

# Save TF-IDF Vectorizer
joblib.dump(tfidf, "saved_models/tfidf_vectorizer.pkl")

print("\nTF-IDF Vectorizer saved successfully!")