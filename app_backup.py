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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 30px;
}

.result-real {
    padding: 28px;
    border-radius: 16px;
    background: #064e3b;
    border: 2px solid #10b981;
    text-align: center;
    font-size: 30px;
    font-weight: 700;
}

.result-fake {
    padding: 28px;
    border-radius: 16px;
    background: #7f1d1d;
    border: 2px solid #ef4444;
    text-align: center;
    font-size: 30px;
    font-weight: 700;
}

.info-card {
    padding: 20px;
    border-radius: 14px;
    background: #1f2937;
    margin-top: 15px;
}

.disclaimer {
    padding: 15px;
    border-radius: 10px;
    background: #292313;
    border: 1px solid #8a7225;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_paths = [
        "saved_models/random_forest.pkl",
        "saved_models/random_forest_model.pkl",
        "models/random_forest.pkl",
        "models/random_forest_model.pkl"
    ]

    vectorizer_paths = [
        "saved_models/tfidf_vectorizer.pkl",
        "saved_models/tfidf.pkl",
        "models/tfidf_vectorizer.pkl",
        "models/tfidf.pkl"
    ]

    model = None
    vectorizer = None

    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break

    for path in vectorizer_paths:
        if os.path.exists(path):
            vectorizer = joblib.load(path)
            break

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Model:** Random Forest")

    st.write("**Feature Extraction:** TF-IDF")

    st.write("**Task:** Binary Text Classification")

    st.write("**Classes:** Fake / Real")

    st.write("**Test Accuracy:** 99.85%")

    st.divider()

    st.header("📊 Dataset")

    st.write("**Records:** 44,689")

    st.write("**Classes:** 2")

    st.write("• Fake News")
    st.write("• Real News")

    st.divider()

    st.info(
        "This system predicts whether a news article "
        "is likely to be REAL or FAKE based on patterns "
        "learned from the training dataset."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'NLP-based misinformation classification using '
    'TF-IDF and Random Forest'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL CHECK
# ============================================================

if model is None or vectorizer is None:

    st.error("❌ Required model files could not be found.")

    st.info("""
    Please make sure the trained Random Forest model
    and TF-IDF vectorizer are available in the project.
    """)

    st.stop()


# ============================================================
# INPUT SECTION
# ============================================================

st.divider()

st.subheader("📝 Analyze a News Article")

st.write(
    "Paste a complete news article below and click "
    "**Analyze News**."
)

news_text = st.text_area(
    "News Article",
    height=260,
    placeholder=(
        "Example:\n\n"
        "The government announced a new policy today "
        "following a cabinet meeting..."
    )
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    analyze = st.button(
        "🔍 Analyze News",
        use_container_width=True
    )

with col2:

    clear = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


if clear:

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not news_text.strip():

        st.warning(
            "⚠️ Please enter a news article first."
        )

    else:

        with st.spinner(
            "🔄 Analyzing the news article..."
        ):

            processed_text = preprocess_text(
                news_text
            )

            features = vectorizer.transform(
                [processed_text]
            )

            prediction = model.predict(
                features
            )[0]

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    features
                )[0]

                confidence = (
                    float(np.max(probabilities))
                    * 100
                )

            else:

                confidence = 0.0


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("🔎 Prediction Result")

        if int(prediction) == 1:

            st.markdown(
                '<div class="result-real">'
                '✅ REAL NEWS'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-fake">'
                '❌ FAKE NEWS'
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.subheader("📊 Prediction Confidence")

        st.progress(
            min(confidence / 100, 1.0)
        )

        st.write(
            f"**Confidence: {confidence:.2f}%**"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()

        st.subheader("📈 Model Performance")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Model",
                "Random Forest"
            )

        with m2:

            st.metric(
                "Feature Extraction",
                "TF-IDF"
            )

        with m3:

            st.metric(
                "Test Accuracy",
                "99.85%"
            )


        # ====================================================
        # HOW IT WORKS
        # ====================================================

        st.divider()

        st.subheader("⚙️ How the System Works")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown("### 1️⃣ Input")

            st.write(
                "User enters a news article."
            )

        with c2:

            st.markdown("### 2️⃣ Cleaning")

            st.write(
                "The text is cleaned and normalized."
            )

        with c3:

            st.markdown("### 3️⃣ TF-IDF")

            st.write(
                "The text is converted into numerical features."
            )

        with c4:

            st.markdown("### 4️⃣ Prediction")

            st.write(
                "Random Forest predicts REAL or FAKE."
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            '<div class="disclaimer">'
            '⚠️ <b>Important:</b> This system provides an '
            'AI-based prediction. The result should not be '
            'considered a definitive fact-check. Important '
            'information should be verified using reliable '
            'sources.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📚 About the Project")

st.write(
    "This project uses Natural Language Processing (NLP) "
    "techniques to classify news articles as REAL or FAKE. "
    "The system uses TF-IDF for text feature extraction "
    "and a Random Forest classifier for prediction."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Fake News Detection System | '
    'TF-IDF + Random Forest | NLP Project'
    '</div>',
    unsafe_allow_html=True
)