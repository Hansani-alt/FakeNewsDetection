import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

# Convert tokens to text if needed
if "tokens" in data.columns:
    data["text"] = data["tokens"].astype(str)

# Features and labels
X = data["text"]
y = data["label"]

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "saved_models/random_forest_model.pkl")
joblib.dump(vectorizer, "saved_models/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
