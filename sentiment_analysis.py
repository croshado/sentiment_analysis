# Import necessary libraries
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from textblob import TextBlob  # Sentiment analysis tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset (Assuming it's a CSV file)
df = pd.read_csv('amazon_reviews.csv')

# Check for missing values and fill with an empty string
df['reviews.title'].fillna('', inplace=True)
df['reviews.text'].fillna('', inplace=True)

# Define text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):  # Ensure input is a string
        text = text.lower()  # Convert to lowercase
        text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])  # Remove punctuation & digits
        words = text.split()  # Tokenize words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)
    else:
        return ''  # Return empty string for missing values

# Apply preprocessing
df['cleaned_title'] = df['reviews.title'].apply(preprocess_text)
df['cleaned_text'] = df['reviews.text'].apply(preprocess_text)

# Combine title and text into one feature
df['combined_review'] = df['cleaned_title'] + ' ' + df['cleaned_text']

# Use TextBlob to generate sentiment labels (0 = Negative, 1 = Positive)
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity  # Compute polarity (-1 to 1)
    return 1 if polarity > 0 else 0  # 1 = Positive, 0 = Negative

df['sentiment'] = df['combined_review'].apply(get_sentiment)  # Generate labels

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['combined_review'])
y = df['sentiment']  # Labels from sentiment analysis

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Hyperparameter tuning using GridSearchCV
params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), params)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate best model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)

print("\nBest Model Evaluation:")
print(f"Accuracy: {accuracy_best:.2f}")
print(f"Precision: {precision_best:.2f}")
print(f"Recall: {recall_best:.2f}")