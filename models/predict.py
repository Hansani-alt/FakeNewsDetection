import joblib

# Load saved model and vectorizer
model = joblib.load("saved_models/random_forest_model.pkl")
vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")

while True:
    news = input("\nEnter News (type 'exit' to quit): ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        print("Prediction: REAL NEWS")
    else:
        print("Prediction: FAKE NEWS")