# Fake News Detection System Using Natural Language Processing

## Module Information

- **Module:** Natural Language Processing (CCS3356)
- **Campus:** Sri Lanka Technology Campus (SLTC)
- **Group Number:** 14

## Group Members

1. **Supun Saranga** - CIT-24-01-0552
2. **Dilmi Wijenayaka** - CIT-24-01-0206
3. **R.A. Hansani Saumya** - CIT-24-01-0542

---

## Project Overview

This project focuses on identifying and classifying fake news articles using Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques.

The system evaluates three different models and provides a Streamlit-based web application for users to classify news articles as **REAL** or **FAKE**.

---

## Dataset Information

The project uses the **Fake and Real News Dataset**.

- **Original Dataset Records:** 44,898
- **Processed Dataset Records:** 44,689
- **Classes:** Fake, Real

### Label Mapping

- **0 = FAKE**
- **1 = REAL**

---

## Machine Learning and Deep Learning Models

Three models were implemented and evaluated for fake news classification.

### 1. Random Forest

Random Forest was implemented as a traditional machine learning classifier using TF-IDF features.

**Test Accuracy: 99.85%**

### 2. Support Vector Machine (SVM)

SVM was implemented as a comparison machine learning model using TF-IDF features.

**Test Accuracy: 99.27%**

### 3. BERT

BERT (Bidirectional Encoder Representations from Transformers) was implemented as a transformer-based deep learning approach for text classification.

**Test Accuracy: 99.98%**

---

## Model Accuracy Comparison

| Model | Type | Accuracy |
|---|---|---:|
| Random Forest | Machine Learning | 99.85% |
| SVM | Machine Learning | 99.27% |
| BERT | Transformer / Deep Learning | 99.98% |

**BERT achieved the highest reported accuracy of 99.98% among the three evaluated models.**

---

## Web Application

A **Streamlit-based web application** was developed to provide an interactive interface for fake news classification.

Users can:

- Select a machine learning or deep learning model
- Enter a news article
- Predict whether the article is **REAL** or **FAKE**
- View prediction confidence
- Compare model accuracy

---

## Screenshots

### BERT Prediction

![BERT Prediction](screenshots/BERT%20prediction.png)

### SVM Prediction

![SVM Prediction](screenshots/SVM%20prediction.png)

### Random Forest Prediction

![Random Forest Prediction](screenshots/Random%20Forest%20prediction.png)

### Model Accuracy Comparison

![Model Accuracy Comparison](screenshots/Model%20Accuracy%20Comparison%20chart.png)

---

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Scikit-learn
- BERT
- Transformers
- Streamlit
- TF-IDF
- Random Forest
- Support Vector Machine (SVM)

---

## Project Structure

```text
FakeNewsDetection/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│   └── preprocessed_news.csv
│
├── models/
│   ├── random_forest.py
│   ├── svm_model.py
│   ├── bert_model.py
│   ├── predict.py
│   └── predict_svm.py
│
├── saved_models/
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bert_model/
│
├── screenshots/
│   ├── BERT prediction.png
│   ├── SVM prediction.png
│   ├── Random Forest prediction.png
│   └── Model Accuracy Comparison chart.png
│
├── src/
│
├── app.py
├── requirements.txt
└── README.md
```
## How the System Works

1. **Input** – The user enters a news article.
2. **Preprocessing** – The text is cleaned and prepared for classification.
3. **Feature Extraction** – TF-IDF is used for the traditional machine learning models.
4. **Model Prediction** – The selected model classifies the article as REAL or FAKE.
5. **Confidence Score** – The system displays the prediction confidence.

---

## System Limitation

The system classifies news articles based on patterns learned from the training dataset. Therefore, the prediction should be considered an automated classification result and not a final verification of factual truth.

---

## Conclusion

The project demonstrates the application of Natural Language Processing, Machine Learning, and Deep Learning techniques for automated fake news classification.

Among the three evaluated models, **BERT achieved the highest reported accuracy of 99.98%**, followed by **Random Forest with 99.85%** and **SVM with 99.27%**.
