import pandas as pd
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

# Load dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

# Convert tokens to text
data["processed_text"] = data["tokens"].apply(
    lambda x: " ".join(eval(x)) if isinstance(x, str) else " ".join(x)
)

print("Dataset Shape:")
print(data.shape)

print("\nProcessed Text:")
print(data["processed_text"].head())

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["processed_text"],
    data["label"],
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(train_texts))
print("Testing Samples:", len(test_texts))

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

print("\nTokenization Completed!")

# Create HuggingFace Dataset
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

print("\nDatasets Created Successfully!")

# Load BERT classification model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

print("\nBERT Classification Model Loaded!")

training_args = TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("\nTrainer Ready!")

print("\nTraining BERT Model...")

trainer.train()

print("\nTraining Completed!")

#Evaluation
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict
predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_labels.values

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

trainer.save_model("saved_models/bert_model")

print("\nBERT Model Saved Successfully!")