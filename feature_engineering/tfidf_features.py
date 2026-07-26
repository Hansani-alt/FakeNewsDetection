import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

print(data.head())

print("\nDataset Shape:")
print(data.shape)
# Convert tokens list back to text
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

from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

