import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
 

# Load the preprocessed dataset
df = pd.read_csv("data/preprocessed_news.csv")

print("Dataset loaded successfully!")
print(df.head())

print("\nColumns:")
print(df.columns)
import ast

# Convert token strings to Python lists
sentences = df["tokens"].apply(ast.literal_eval)

print("\nFirst Tokenized Sentence:")
print(sentences.iloc[0])
# Train Word2Vec Model
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

print("\nWord2Vec model trained successfully!")
import os

os.makedirs("saved_models", exist_ok=True)

word2vec_model.save("saved_models/word2vec_model.model")

print("Word2Vec Model Saved Successfully!")
# Function to create average Word2Vec vector
def document_vector(doc):
    words = [word for word in doc if word in word2vec_model.wv]

    if len(words) == 0:
        return np.zeros(100)

    return np.mean(word2vec_model.wv[words], axis=0)


# Create vectors for all documents
X = np.array([document_vector(doc) for doc in sentences])

print("\nWord2Vec Feature Matrix Shape:")
print(X.shape)