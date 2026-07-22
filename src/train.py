import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#Load Data
df = pd.read_csv("data/processed/processed_reviews.csv")

#Combine title and review text
df["Combined Text"] = (df["Review Title"].fillna("") + " " + df['Review Text'].fillna(""))

#Features and labels
X = df["Combined Text"]
y = df["Sentiment"]

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Train Model
model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

#Predictions
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))