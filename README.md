 # Fake News Detection System Using Natural Language Processing

## Module Information

* **Module:** Natural Language Processing (CCS3356)
* **Campus:** Sri Lanka Technology Campus (SLTC)
* **Group Number:** 14

## Group Members

1. **Supun Saranga** - CIT-24-01-0552
2. **Dilmi Wijenayaka** - CIT-24-01-0206
3. **R.A. Hansani Saumya** - CIT-24-01-0542

## Project Overview

This project focuses on identifying and classifying fake news articles using Natural Language Processing (NLP) and Machine Learning / Deep Learning techniques.

## Dataset Information

* **Name:** Fake and Real News Dataset (Kaggle)
* **Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
* **Total Records:** 44,898
* **Classes:** Fake, Real
* 
## Machine Learning Models

The system uses three machine learning/deep learning models for fake news classification:

- Random Forest
- Support Vector Machine (SVM)
- BERT

## Model Accuracy Comparison

| Model | Accuracy |
|---|---:|
| Random Forest | 99.78% |
| SVM | 99.22% |
| BERT | 99.98% |

BERT achieved the highest accuracy among the three models with an accuracy of 99.98%.

## Web Application

A Streamlit-based web application was developed to allow users to:

- Select a machine learning model
- Enter a news article
- Predict whether the article is REAL or FAKE
- View prediction confidence
- Compare model accuracy

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
