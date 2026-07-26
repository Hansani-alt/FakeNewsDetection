import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

print("Dataset Shape:")
print(data.shape)

# Convert tokens to sentence
data["processed_text"] = data["tokens"].apply(
    lambda x: " ".join(eval(x)) if isinstance(x, str) else " ".join(x)
)

print("\nProcessed Text:")
print(data["processed_text"].head())

# Load pretrained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

print("\nBERT Loaded Successfully!")

# Sample sentence
sample = data["processed_text"][0]

inputs = tokenizer(
    sample,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.last_hidden_state[:, 0, :]

print("\nEmbedding Shape:")
print(embedding.shape)