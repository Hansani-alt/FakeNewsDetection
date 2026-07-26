import pandas as pd
import numpy as np
import ast

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Load dataset
df = pd.read_csv("data/preprocessed_news.csv")

# Convert token strings back to list
sentences = df["tokens"].apply(ast.literal_eval)

# Labels
y = df["label"]
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
def document_vector(doc):
    words = [word for word in doc if word in word2vec_model.wv]

    if len(words) == 0:
        return np.zeros(100)

    return np.mean(word2vec_model.wv[words], axis=0)

X = np.array([document_vector(doc) for doc in sentences])
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# Train SVM
svm_model = SVC(kernel="linear")

print("\nTraining SVM Model...")

svm_model.fit(X_train, y_train)

print("SVM Model Trained Successfully!")
# Predictions
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))
import joblib
import os

os.makedirs("saved_models", exist_ok=True)

joblib.dump(svm_model, "saved_models/svm_model.pkl")

print("\nSVM Model Saved Successfully!")