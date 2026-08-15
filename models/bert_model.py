import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv("dataset/preprocessed_news.csv")

# Use a balanced subset for CPU-based BERT training
fake_data = data[data["label"] == 0].sample(n=3000, random_state=42)
real_data = data[data["label"] == 1].sample(n=3000, random_state=42)

data = pd.concat([fake_data, real_data])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("BERT Dataset Shape:", data.shape)

print("Dataset Shape:", data.shape)


# ============================================================
# 2. CONVERT TOKENS TO TEXT
# ============================================================

def convert_tokens(x):
    if isinstance(x, str):
        try:
            return " ".join(eval(x))
        except:
            return x
    return " ".join(x)


data["processed_text"] = data["tokens"].apply(convert_tokens)


# Remove empty rows
data = data[
    data["processed_text"].notna() &
    (data["processed_text"].str.strip() != "")
].reset_index(drop=True)


# ============================================================
# 3. CHECK LABELS
# ============================================================

print("\nLabel Distribution:")
print(data["label"].value_counts())

print("\nLabel Meaning:")
print("0 = FAKE")
print("1 = REAL")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["processed_text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print("\nTraining Samples:", len(train_texts))
print("Testing Samples:", len(test_texts))


# ============================================================
# 5. LOAD BERT TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ============================================================
# 6. TOKENIZATION
# ============================================================

print("\nTokenizing training data...")

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenizing testing data...")

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenization completed!")


# ============================================================
# 7. CREATE DATASETS
# ============================================================

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels.tolist()
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels.tolist()
})

print("\nDatasets created successfully!")


# ============================================================
# 8. LOAD BERT CLASSIFICATION MODEL
# ============================================================

print("\nLoading BERT classification model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label={
        0: "FAKE",
        1: "REAL"
    },
    label2id={
        "FAKE": 0,
        "REAL": 1
    }
)

print("BERT model loaded successfully!")


# ============================================================
# 9. TRAINING SETTINGS
# ============================================================

training_args = TrainingArguments(
    output_dir="bert_results",
    
    eval_strategy="epoch",
    save_strategy="epoch",

    num_train_epochs=1,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,
    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=False,

    report_to="none"
)


# ============================================================
# 10. TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n")
print("=" * 60)
print("             TRAINING BERT MODEL")
print("=" * 60)

trainer.train()

print("\nTraining completed!")


# ============================================================
# 12. EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("             EVALUATING BERT MODEL")
print("=" * 60)

predictions = trainer.predict(test_dataset)

y_pred = np.argmax(
    predictions.predictions,
    axis=1
)

y_true = test_labels.to_numpy()


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["FAKE", "REAL"],
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 16. PREDICTED LABEL DISTRIBUTION
# ============================================================

print("\nPredicted Label Distribution:")
print(
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
)


# ============================================================
# 17. SAVE FINAL MODEL
# ============================================================

print("\n")
print("=" * 60)
print("             SAVING FINAL MODEL")
print("=" * 60)

trainer.save_model(
    "saved_models/bert_model"
)

tokenizer.save_pretrained(
    "saved_models/bert_model"
)

print("\nBERT MODEL SAVED SUCCESSFULLY!")


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("                 FINAL SUMMARY")
print("=" * 60)

print(f"Dataset Size       : {len(data)}")
print(f"Training Samples   : {len(train_texts)}")
print(f"Testing Samples    : {len(test_texts)}")
print(f"Test Accuracy      : {accuracy * 100:.2f}%")

print("\nLabel Mapping:")
print("0 = FAKE")
print("1 = REAL")

print("\nBERT training and evaluation completed!")