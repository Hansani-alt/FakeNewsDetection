import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/preprocessed_news.csv")

if "tokens" in data.columns:
    data["text"] = data["tokens"].astype(str)

X = data["text"]
y = data["label"]


# ==========================================
# LOAD SAVED TF-IDF VECTORIZER
# ==========================================

vectorizer = joblib.load(
    "saved_models/tfidf_vectorizer.pkl"
)

X_tfidf = vectorizer.transform(X)


# ==========================================
# SAME TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# LOAD SAVED RANDOM FOREST MODEL
# ==========================================

model = joblib.load(
    "saved_models/random_forest_model.pkl"
)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("MODEL EVALUATION")
print("====================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "REAL"]
)

disp.plot()

plt.title("Fake News Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion Matrix saved successfully!")