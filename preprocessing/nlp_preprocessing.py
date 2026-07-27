import pandas as pd

# Load cleaned dataset
data = pd.read_csv("dataset/clean_news.csv")

# Remove missing text values
data = data.dropna(subset=["text"])

# Make sure text column is string
data["text"] = data["text"].astype(str)

print(data.head())

print("\nDataset Shape:")
print(data.shape)

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize

# Tokenization
data["tokens"] = data["text"].apply(word_tokenize)

print("\nTokenization Completed!")
print(data["tokens"].head())

from nltk.corpus import stopwords

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Remove stopwords
data["tokens"] = data["tokens"].apply(
    lambda words: [word for word in words if word.lower() not in stop_words]
)

print("\nStopword Removal Completed!")
print(data["tokens"].head())

from nltk.stem import WordNetLemmatizer

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply lemmatization
data["tokens"] = data["tokens"].apply(
    lambda words: [lemmatizer.lemmatize(word) for word in words]
)

print("\nLemmatization Completed!")
print(data["tokens"].head())

# Save preprocessed dataset
data.to_csv("dataset/preprocessed_news.csv", index=False)

print("\nPreprocessed dataset saved successfully!")