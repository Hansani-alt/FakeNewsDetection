import joblib
import numpy as np
import re

from gensim.models import Word2Vec

# Load trained SVM model
svm_model = joblib.load("saved_models/svm_model.pkl")

# Load trained Word2Vec model
word2vec_model = Word2Vec.load("saved_models/word2vec_model.model")


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.split()


def document_vector(doc):
    words = [word for word in doc if word in word2vec_model.wv]

    if len(words) == 0:
        return np.zeros(100)

    return np.mean(word2vec_model.wv[words], axis=0)


news = input("Enter News : ")

tokens = preprocess(news)

vector = document_vector(tokens).reshape(1, -1)

prediction = svm_model.predict(vector)

if prediction[0] == 1:
    print("\nPrediction : REAL NEWS")
else:
    print("\nPrediction : FAKE NEWS")