import streamlit as st
import joblib
import numpy as np
import re

# -----------------------------
# Load Random Forest model
# -----------------------------
rf_model = joblib.load("saved_models/random_forest_model.pkl")
tfidf_vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")


# -----------------------------
# Load SVM model + Word2Vec
# -----------------------------
svm_model = joblib.load("saved_models/svm_model.pkl")

from gensim.models import Word2Vec

word2vec_model = Word2Vec.load(
    "saved_models/word2vec_model.model"
)


# -----------------------------
# SVM preprocessing
# -----------------------------
def preprocess_svm(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()


def document_vector(doc):
    words = [
        word for word in doc
        if word in word2vec_model.wv
    ]

    if len(words) == 0:
        return np.zeros(100)

    return np.mean(
        word2vec_model.wv[words],
        axis=0
    )


# -----------------------------
# BERT imports
# -----------------------------
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)
import torch


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📰 Fake News Detection System")

st.write(
    "Select a machine learning model and enter a news article "
    "to check whether it is REAL or FAKE."
)


# Model selection
model_choice = st.selectbox(
    "Select Model:",
    [
        "Random Forest",
        "SVM",
        "BERT"
    ]
)


# News input
news = st.text_area(
    "Enter News Article:",
    height=180
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        # =============================
        # RANDOM FOREST
        # =============================
        if model_choice == "Random Forest":

            news_vector = tfidf_vectorizer.transform([news])

            prediction = rf_model.predict(news_vector)[0]

            probabilities = rf_model.predict_proba(
                news_vector
            )[0]

            confidence = max(probabilities) * 100


        # =============================
        # SVM
        # =============================
        elif model_choice == "SVM":

            tokens = preprocess_svm(news)

            vector = document_vector(tokens).reshape(1, -1)

            prediction = svm_model.predict(vector)[0]

            # SVM probability may not be available
            if hasattr(svm_model, "predict_proba"):

                probabilities = svm_model.predict_proba(vector)[0]

                confidence = max(probabilities) * 100

            else:

                confidence = 100.0


        # =============================
        # BERT
        # =============================
        else:

            tokenizer = BertTokenizer.from_pretrained(
                "saved_models/bert_model"
            )

            bert_model = BertForSequenceClassification.from_pretrained(
                "saved_models/bert_model"
            )

            bert_model.eval()

            inputs = tokenizer(
                news,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():

                outputs = bert_model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )[0]

            prediction = torch.argmax(
                probabilities
            ).item()

            confidence = probabilities[
                prediction
            ].item() * 100


        # =============================
        # DISPLAY RESULT
        # =============================

        st.subheader(
            f"Model Used: {model_choice}"
        )

        if prediction == 1:

            st.success("✅ REAL NEWS")

        else:

            st.error("❌ FAKE NEWS")


        st.info(
            f"Prediction Confidence: {confidence:.2f}%"
        )
    st.subheader("📊 Model Accuracy Comparison")

    accuracy_data = {
        "Model": ["BERT", "Random Forest", "SVM"],
        "Accuracy": [99.98, 99.78, 99.22]
}

    st.bar_chart(
        accuracy_data,
        x="Model",
        y="Accuracy"
)

    st.success("🏆 Best Performing Model: BERT — 99.98% Accuracy")