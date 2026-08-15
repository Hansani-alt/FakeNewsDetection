import streamlit as st
import joblib
import os
import re
import numpy as np

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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_paths = [
        "saved_models/random_forest_model.pkl",
        "saved_models/random_forest.pkl",
        "models/random_forest_model.pkl",
        "models/random_forest.pkl"
    ]

    vectorizer_paths = [
        "saved_models/tfidf_vectorizer.pkl",
        "saved_models/tfidf.pkl",
        "models/tfidf_vectorizer.pkl",
        "models/tfidf.pkl"
    ]

    model = None
    vectorizer = None

    # Load Random Forest model
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break

    # Load TF-IDF vectorizer
    for path in vectorizer_paths:
        if os.path.exists(path):
            vectorizer = joblib.load(path)
            break

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered news classification using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MODEL CHECK
# ============================================================

if model is None or vectorizer is None:

    st.error("❌ Model files could not be found.")

    st.info("""
    Please make sure these files exist:

    saved_models/random_forest_model.pkl
    saved_models/tfidf_vectorizer.pkl
    """)

    st.stop()


# ============================================================
# NEWS INPUT
# ============================================================

st.subheader("📝 Enter News Article")

st.write("Paste the news article below:")

news_text = st.text_area(
    "",
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
# PREDICTION
# ============================================================

if st.button("🔍 Analyze News", use_container_width=True):

    if news_text.strip() == "":

        st.warning("⚠️ Please enter a news article first.")

    else:

        with st.spinner("Analyzing news article..."):

            # Preprocess
            processed_text = preprocess_text(news_text)

            # TF-IDF transformation
            features = vectorizer.transform([processed_text])

            # Prediction
            prediction = model.predict(features)[0]

            # Confidence
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(features)[0]

                confidence = float(np.max(probabilities)) * 100

            else:

                confidence = 0.0


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader("🔎 Prediction Result")

        if int(prediction) == 1:

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

            st.metric(
                "Selected Model",
                "Random Forest"
            )

        with col2:

            st.metric(
                "Feature Extraction",
                "TF-IDF"
            )

        with col3:

            st.metric(
                "Test Accuracy",
                "99.85%"
            )


        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.divider()

        st.subheader("📈 Model Accuracy Comparison")

        comparison_path = "screenshots/model_comparison.png"

        if os.path.exists(comparison_path):

            st.image(
                comparison_path,
                caption="Comparison of Machine Learning Models",
                use_container_width=True
            )

        else:

            st.warning("Model comparison graph not found.")


        # ====================================================
        # MODEL COMPARISON TABLE
        # ====================================================

        st.subheader("📋 Performance Comparison")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🌲 Random Forest",
                "99.85%"
            )

        with col2:

            st.metric(
                "SVM",
                "99.27%"
            )

        st.success(
            "Random Forest achieved the highest test accuracy and was selected as the final model."
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.divider()

        st.subheader("🔢 Confusion Matrix")

        confusion_path = "screenshots/confusion_matrix.png"

        if os.path.exists(confusion_path):

            st.image(
                confusion_path,
                caption="Random Forest Confusion Matrix",
                use_container_width=True
            )

        else:

            st.warning("Confusion matrix image not found.")


        # ====================================================
        # EVALUATION SUMMARY
        # ====================================================

        st.subheader("📊 Evaluation Summary")

        evaluation_data = {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ],
            "Random Forest": [
                "99.85%",
                "1.00",
                "1.00",
                "1.00"
            ],
            "SVM": [
                "99.27%",
                "0.99",
                "0.99",
                "0.99"
            ]
        }

        st.table(evaluation_data)


        # ====================================================
        # HOW IT WORKS
        # ====================================================

        st.divider()

        st.subheader("⚙️ How It Works")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown("### 1️⃣ Input")

            st.write(
                "The user enters a news article into the system."
            )

        with c2:

            st.markdown("### 2️⃣ Preprocessing")

            st.write(
                "The text is cleaned by converting it to lowercase and removing unnecessary characters."
            )

        with c3:

            st.markdown("### 3️⃣ TF-IDF")

            st.write(
                "The cleaned text is converted into numerical TF-IDF features."
            )

        with c4:

            st.markdown("### 4️⃣ Prediction")

            st.write(
                "The Random Forest classifier predicts whether the article is REAL or FAKE."
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
            st.info("📊 TF-IDF")

        with tech4:
            st.info("🌲 Random Forest")


        # ====================================================
        # LIMITATION
        # ====================================================

        st.divider()

        st.subheader("⚠️ System Limitation")

        st.write(
            "The system classifies news based on patterns learned from the training dataset. "
            "Therefore, the prediction should be considered as an automated classification result "
            "and not as a final verification of factual truth."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fake News Detection System | TF-IDF + Random Forest | NLP Project"
)