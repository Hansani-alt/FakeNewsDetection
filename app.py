import streamlit as st
import joblib
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #b0b0b0;
    margin-bottom: 40px;
}

.result-real {
    padding: 30px;
    border-radius: 15px;
    background-color: #064e3b;
    border: 2px solid #10b981;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.result-fake {
    padding: 30px;
    border-radius: 15px;
    background-color: #7f1d1d;
    border: 2px solid #ef4444;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1f2937;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

RF_MODEL_PATH = "saved_models/random_forest_model.pkl"
SVM_MODEL_PATH = "saved_models/svm_model.pkl"
TFIDF_PATH = "saved_models/tfidf_vectorizer.pkl"
BERT_PATH = "saved_models/bert_model"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():
    rf_model = None
    svm_model = None
    vectorizer = None
    bert_tokenizer = None
    bert_model = None

    # Random Forest
    if os.path.exists(RF_MODEL_PATH):
        rf_model = joblib.load(RF_MODEL_PATH)

    # SVM
    if os.path.exists(SVM_MODEL_PATH):
        svm_model = joblib.load(SVM_MODEL_PATH)

    # TF-IDF
    if os.path.exists(TFIDF_PATH):
        vectorizer = joblib.load(TFIDF_PATH)

    # BERT
    if os.path.isdir(BERT_PATH):
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
            bert_model.eval()
        except Exception as e:
            st.warning(f"BERT model could not be loaded: {e}")

    return rf_model, svm_model, vectorizer, bert_tokenizer, bert_model


rf_model, svm_model, vectorizer, bert_tokenizer, bert_model = load_all_models()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered news classification using NLP, Machine Learning and BERT</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MODEL SELECTION
# ============================================================

available_models = []

if bert_model is not None and bert_tokenizer is not None:
    available_models.append("BERT")

if rf_model is not None and vectorizer is not None:
    available_models.append("Random Forest")

if svm_model is not None and vectorizer is not None:
    available_models.append("SVM")

if not available_models:
    st.error("No trained models could be loaded.")
    st.stop()

st.subheader("🤖 Select Model")

selected_model = st.selectbox(
    "Choose a model for prediction:",
    available_models,
    index=0
)

st.caption(
    "BERT is the final selected model because it achieved the highest reported test accuracy of 99.98%."
)


# ============================================================
# NEWS INPUT
# ============================================================

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste the news article below:",
    height=250,
    placeholder="Example: The government announced a new policy today following a cabinet meeting..."
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_tfidf(model, text):
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])

    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities)) * 100
    else:
        confidence = 0.0

    return prediction, confidence


def predict_bert(text):
    # Keep the original article text for BERT.
    # The transformer tokenizer performs its own preprocessing/tokenization.
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    prediction = int(torch.argmax(probabilities).item())
    confidence = float(torch.max(probabilities).item()) * 100

    return prediction, confidence


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Analyze News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:
        with st.spinner(f"Analyzing news using {selected_model}..."):

            if selected_model == "BERT":
                prediction, confidence = predict_bert(news_text)
                feature_extraction = "BERT Tokenization"
                test_accuracy = "99.98%"

            elif selected_model == "Random Forest":
                prediction, confidence = predict_tfidf(rf_model, news_text)
                feature_extraction = "TF-IDF"
                test_accuracy = "99.85%"

            else:
                prediction, confidence = predict_tfidf(svm_model, news_text)
                feature_extraction = "TF-IDF"
                test_accuracy = "99.27%"


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()
        st.subheader("🔎 Prediction Result")

        if prediction == 1:
            st.markdown(
                '<div class="result-real">✅ REAL NEWS</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-fake">❌ FAKE NEWS</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.subheader("📊 Prediction Confidence")
        st.progress(min(confidence / 100, 1.0))
        st.write(f"**Confidence: {confidence:.2f}%**")


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()
        st.subheader("🤖 Model Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Selected Model", selected_model)

        with col2:
            st.metric("Feature Extraction", feature_extraction)

        with col3:
            st.metric("Test Accuracy", test_accuracy)


        # ====================================================
        # MODEL ACCURACY COMPARISON
        # ====================================================

        st.divider()
        st.subheader("📈 Model Accuracy Comparison")

        comparison_data = pd.DataFrame({
            "Model": ["BERT", "Random Forest", "SVM"],
            "Accuracy": [99.98, 99.85, 99.27]
        })

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(comparison_data["Model"], comparison_data["Accuracy"])

        ax.set_title("Fake News Detection - Model Accuracy Comparison")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(98.5, 100.1)

        for bar, value in zip(bars, comparison_data["Accuracy"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold"
            )

        st.pyplot(fig)
        plt.close(fig)


        # ====================================================
        # PERFORMANCE COMPARISON
        # ====================================================

        st.subheader("📋 Performance Comparison")

        performance_data = pd.DataFrame({
            "Model": ["BERT", "Random Forest", "SVM"],
            "Type": [
                "Transformer / Deep Learning",
                "Machine Learning",
                "Machine Learning"
            ],
            "Accuracy": ["99.98%", "99.85%", "99.27%"]
        })

        st.table(performance_data)

        st.success(
            "BERT achieved the highest reported test accuracy of 99.98% "
            "and was selected as the final model."
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.divider()
        st.subheader("🔢 Random Forest Confusion Matrix")

        confusion_path = "screenshots/confusion_matrix.png"

        if os.path.exists(confusion_path):
            st.image(
                confusion_path,
                caption="Random Forest Confusion Matrix",
                use_container_width=True
            )
        else:
            st.info("Confusion matrix image not found.")


        # ====================================================
        # EVALUATION SUMMARY
        # ====================================================

        st.subheader("📊 Evaluation Summary")

        evaluation_data = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "BERT": ["99.98%", "N/A", "N/A", "N/A"],
            "Random Forest": ["99.85%", "1.00", "1.00", "1.00"],
            "SVM": ["99.27%", "0.99", "0.99", "0.99"]
        })

        st.table(evaluation_data)

        st.caption(
            "BERT precision, recall and F1-score are shown as N/A because the "
            "provided BERT evaluation results only report test accuracy."
        )


        # ====================================================
        # HOW IT WORKS
        # ====================================================

        st.divider()
        st.subheader("⚙️ How It Works")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("### 1️⃣ Input")
            st.write("The user enters a news article into the system.")

        with c2:
            st.markdown("### 2️⃣ Preprocessing")
            st.write(
                "The text is cleaned and prepared for classification. "
                "Traditional models use the project's text preprocessing pipeline."
            )

        with c3:
            st.markdown("### 3️⃣ Feature Extraction")
            st.write(
                "TF-IDF features are used for Random Forest and SVM, "
                "while BERT uses transformer tokenization."
            )

        with c4:
            st.markdown("### 4️⃣ Prediction")
            st.write(
                "The selected model predicts whether the article is REAL or FAKE."
            )


        # ====================================================
        # PROJECT TECHNOLOGY
        # ====================================================

        st.divider()
        st.subheader("🛠️ Technologies Used")

        tech1, tech2, tech3, tech4 = st.columns(4)

        with tech1:
            st.info("🐍 Python")

        with tech2:
            st.info("📝 NLP")

        with tech3:
            st.info("🤖 BERT")

        with tech4:
            st.info("📊 TF-IDF")


        # ====================================================
        # LIMITATION
        # ====================================================

        st.divider()
        st.subheader("⚠️ System Limitation")

        st.write(
            "The system classifies news based on patterns learned from the "
            "training dataset. Therefore, the prediction should be considered "
            "as an automated classification result and not as a final "
            "verification of factual truth."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake News Detection System | BERT + Random Forest + SVM | NLP Project"
)