import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("dataset/preprocessed_news.csv")

print("Dataset Shape:")
print(data.shape)

# Convert token list to text
data["processed_text"] = data["tokens"].apply(
    lambda x: " ".join(eval(x)) if isinstance(x, str) else " ".join(x)
)

print("\nProcessed Text:")
print(data["processed_text"].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data["processed_text"])
y = data["label"]

print("\nTF-IDF Shape:")
print(X.shape)

from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining Random Forest Model...")

model.fit(X_train, y_train)

print("Training Completed!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
joblib.dump(model, "saved_models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")