import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("dataset/preprocessed_news.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. CONVERT TOKENS TO TEXT
# ============================================================

def convert_tokens(x):

    if isinstance(x, str):
        try:
            tokens = eval(x)
            return " ".join(tokens)
        except:
            return x

    return " ".join(x)


df["processed_text"] = df["tokens"].apply(convert_tokens)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["processed_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining samples:", len(train_texts))
print("Testing samples:", len(test_texts))

print("\nActual test label distribution:")
print(test_labels.value_counts().sort_index())


# ============================================================
# 4. LOAD SAVED BERT MODEL
# ============================================================

print("\nLoading saved BERT model...")

tokenizer = BertTokenizer.from_pretrained(
    "saved_models/bert_model"
)

model = BertForSequenceClassification.from_pretrained(
    "saved_models/bert_model"
)

model.eval()

print("BERT model loaded successfully!")


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_texts(texts):

    predictions = []

    batch_size = 16

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():

            outputs = model(**inputs)

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

        predictions.extend(
            preds.cpu().numpy().tolist()
        )

    return np.array(predictions)


# ============================================================
# 6. PREDICT TEST DATA
# ============================================================

print("\nPredicting test data...")

y_true = test_labels.to_numpy()

y_pred = predict_texts(
    test_texts.tolist()
)


# ============================================================
# 7. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n========================================")
print("           BERT TEST RESULTS")
print("========================================")

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 8. PREDICTED LABEL DISTRIBUTION
# ============================================================

print("\nPredicted label distribution:")
print(
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "FAKE",
            "REAL"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 11. SAMPLE PREDICTIONS
# ============================================================

print("\n========================================")
print("         SAMPLE PREDICTIONS")
print("========================================")

for i in range(10):

    actual = "REAL" if y_true[i] == 1 else "FAKE"
    predicted = "REAL" if y_pred[i] == 1 else "FAKE"

    print("\nArticle:")
    print(test_texts.iloc[i][:300])

    print("Actual   :", actual)
    print("Predicted:", predicted)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("             FINAL SUMMARY")
print("========================================")

print(f"Test Samples : {len(y_true)}")
print(f"Correct      : {(y_true == y_pred).sum()}")
print(f"Incorrect    : {(y_true != y_pred).sum()}")
print(f"Accuracy     : {accuracy * 100:.2f}%")

print("\nTesting completed successfully!")