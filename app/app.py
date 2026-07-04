import streamlit as st
import joblib
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------------------
# Load Saved Model and Vectorizer
# -------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "sentiment_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")

# -------------------------------
# Text Preprocessing
# -------------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_review(review):
    review = review.lower()
    review = review.translate(str.maketrans("", "", string.punctuation))
    review = re.sub(r"\d+", "", review)

    words = review.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("📖 About")

st.sidebar.write("""
This application predicts the sentiment of a product review.

### Sentiments
- 😊 Positive
- 😐 Neutral
- 😞 Negative

### Model Used
- Logistic Regression
- TF-IDF Vectorizer
- Scikit-Learn
""")


# -------------------------------
# Main Page
# -------------------------------

st.title("🛍 AI Product Review Sentiment Analyzer")

st.markdown(
    "Analyze customer reviews using **Machine Learning** and **Natural Language Processing (NLP)**."
)

review = st.text_area("✍ Enter your review:")

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Clean Review
        clean_review = preprocess_review(review)

        # Convert to TF-IDF
        review_vector = vectorizer.transform([clean_review])

        # Predict
        prediction = model.predict(review_vector)

        # Confidence
        probabilities = model.predict_proba(review_vector)

        # Display Prediction
        if prediction[0] == "Positive":
            st.success("😊 Prediction: Positive")

        elif prediction[0] == "Neutral":
            st.info("😐 Prediction: Neutral")

        else:
            st.error("😞 Prediction: Negative")

        # Confidence Scores
        st.subheader("Confidence Scores")

        for label, prob in zip(model.classes_, probabilities[0]):
            st.write(f"**{label}: {prob:.2%}**")
            st.progress(float(prob))