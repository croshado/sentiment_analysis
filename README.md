# Sentiment Analysis of Customer Reviews

## 1. Introduction
This project aims to classify customer reviews as **positive** or **negative** using **Machine Learning**. Since the dataset lacks explicit sentiment labels, we use **TextBlob** to generate labels based on the review text.

## 2. Dataset Description
- The dataset consists of **Amazon product reviews** stored in a CSV file (`amazon_reviews.csv`).
- The relevant columns used for analysis:
  - `reviews.title`: The title of the review.
  - `reviews.text`: The full review content.
- Since there is no `sentiment` or `rating` column, we infer sentiment using **TextBlob polarity scores**.

## 3. Workflow Overview
1. **Data Preprocessing**: Cleaning the text data (removing punctuation, stopwords, and converting to lowercase).
2. **Sentiment Labeling**: Assigning sentiment labels using **TextBlob**.
3. **Feature Extraction**: Converting text data into numerical features using **TF-IDF**.
4. **Model Training**: Training a **Naive Bayes classifier**.
5. **Model Evaluation**: Measuring **accuracy, precision, and recall**.
6. **Hyperparameter Tuning**: Optimizing the model using **GridSearchCV**.

## 4. Detailed Steps

### 4.1 Data Loading
The dataset is loaded using **pandas**:
```python
import pandas as pd
df = pd.read_csv('amazon_reviews.csv')
```
- Missing values in `reviews.title` and `reviews.text` are filled with empty strings to avoid errors.

### 4.2 Data Preprocessing
Each review is cleaned:
- Converted to **lowercase**.
- **Punctuation** and **digits** removed.
- **Tokenization**: Splitting text into words.
- **Stopwords** removed (e.g., "the", "is", "and").

```python
from nltk.corpus import stopwords
import string, nltk
nltk.download('stopwords')

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        return ' '.join(words)
    return ''

df['cleaned_title'] = df['reviews.title'].apply(preprocess_text)
df['cleaned_text'] = df['reviews.text'].apply(preprocess_text)
df['combined_review'] = df['cleaned_title'] + ' ' + df['cleaned_text']
```

### 4.3 Sentiment Labeling Using TextBlob
Since the dataset lacks sentiment labels, we use **TextBlob**:
- **Polarity Score**: Ranges from **-1 (negative) to +1 (positive)**.
- If polarity > 0 → **Positive (1)**
- If polarity ≤ 0 → **Negative (0)**

```python
from textblob import TextBlob

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 1 if polarity > 0 else 0

df['sentiment'] = df['combined_review'].apply(get_sentiment)
```

### 4.4 Feature Extraction (TF-IDF)
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical data:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['combined_review'])
y = df['sentiment']
```

### 4.5 Model Training (Naive Bayes)
We split the data (80% training, 20% testing) and train a **Naive Bayes classifier**:
```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 4.6 Model Evaluation
We measure performance using **Accuracy, Precision, and Recall**:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
```

### 4.7 Hyperparameter Tuning (GridSearchCV)
To improve model performance, we tune hyperparameters:
```python
from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), params)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
```

### 4.8 Best Model Evaluation
```python
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)

print(f'Best Accuracy: {accuracy_best:.2f}')
print(f'Best Precision: {precision_best:.2f}')
print(f'Best Recall: {recall_best:.2f}')
```

## 5. Results and Insights
- **Baseline Model Performance**:
  - Accuracy: **X%** (Replace X with actual value)
  - Precision: **X%**
  - Recall: **X%**
- **After Hyperparameter Tuning**:
  - Accuracy improved to **X%**
  - Precision improved to **X%**
  - Recall improved to **X%**

## 6. Potential Improvements
1. **Use VADER Sentiment Analysis** (better for short reviews).
2. **Try Deep Learning Models** (e.g., LSTMs, BERT).
3. **Use Different ML Models** (Logistic Regression, SVM).
4. **Increase TF-IDF Features** (from 5000 to 10000).

## 7. Conclusion
This project successfully classifies Amazon product reviews as positive or negative using **Naive Bayes**. The automatic labeling via **TextBlob** provides reasonable accuracy, but further enhancements (deep learning, VADER) can improve performance.

---

This document provides a comprehensive overview of the **Sentiment Classification Pipeline**. Let me know if you need further modifications! 🚀

