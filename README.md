# sms-spam-project
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and preprocess the dataset
data = pd.read_csv('path_to_dataset.csv')
data['text'] = data['text'].str.lower()  # Convert text to lowercase
data['text'] = data['text'].str.replace('[^\w\s]', '')  # Remove punctuation

# Step 2: Feature Engineering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection and Training
model = MultinomialNB()  # Naive Bayes Classifier
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
