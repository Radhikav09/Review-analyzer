# 🛍️ Product Review Sentiment Analysis

A Machine Learning project that predicts the sentiment of women's clothing product reviews as **Positive**, **Neutral**, or **Negative** using Natural Language Processing (NLP) and Logistic Regression.

---

## 📌 Project Overview

This project analyzes customer reviews from an e-commerce clothing dataset and classifies each review into one of three sentiment categories:

- 😊 Positive
- 😐 Neutral
- ☹️ Negative

The model was trained using TF-IDF Vectorization and Logistic Regression after performing text preprocessing and cleaning.

---

## 📂 Dataset

**Dataset:** Women's Clothing E-Commerce Reviews

The dataset contains customer reviews along with ratings and other product-related information.

For this project, only the following columns were used:

- Review Text
- Rating

The sentiment labels were created from ratings:

| Rating | Sentiment |
|---------|-----------|
| 4 - 5 | Positive |
| 3 | Neutral |
| 1 - 2 | Negative |

---

## 🚀 Features

- Data Cleaning
- Text Preprocessing
- Stopword Removal
- Stemming using Porter Stemmer
- TF-IDF Vectorization
- Logistic Regression Classification
- Confusion Matrix
- Classification Report
- Confidence Score Prediction

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- NLTK
- Scikit-learn
- Jupyter Notebook

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Select Required Columns
3. Remove Missing Values
4. Create Sentiment Labels
5. Clean Review Text
6. Remove Stopwords
7. Apply Stemming
8. Convert Text to TF-IDF Features
9. Split Dataset into Training and Testing Sets
10. Train Logistic Regression Model
11. Evaluate Model Performance
12. Predict Sentiment for New Reviews

---

## 📈 Model Performance

**Algorithm:** Logistic Regression

**Vectorizer:** TF-IDF

**Accuracy:** **~82%**

The model performs well for Positive and Negative reviews while Neutral reviews are comparatively more challenging due to class imbalance in the dataset.

---

## 💬 Example Prediction

**Input**

```
I absolutely love this dress. Excellent quality.
```

**Prediction**

```
Positive
```

**Confidence**

```
Positive : 90.98%
Neutral  : 4.66%
Negative : 4.36%
```

---

## 📁 Project Structure

```
Product-Review-Sentiment-Analysis/
│
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv
│
├── notebook/
│   └── sentiment_analysis.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔮 Future Improvements

- Replace Stemming with Lemmatization
- Experiment with Support Vector Machine (SVM)
- Build an interactive Streamlit web application
- Deploy the project online
- Improve Neutral sentiment classification

---

## 👩‍💻 Author

**Radhika Varma**

GitHub: *Add your GitHub profile link here*

---

## ⭐ If you found this project helpful, consider giving it a star!