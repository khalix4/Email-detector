import numpy as np
import pandas as pd
import pickle
import re
import string
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from xgboost import XGBClassifier
from bs4 import BeautifulSoup
import tldextract

# --- Text cleaning ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_words)

# --- Extract URLs ---
def extract_urls(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    links = [a.get('href') for a in soup.find_all('a', href=True)]
    raw_links = re.findall(r'\b(?:https?|ftp|hxxp)://[^\s"<>\]]+', text)
    all_urls = list(set(links + raw_links))
    return " ".join(filter(None, all_urls))

# --- Validate URL format ---
def is_valid_url(url):
    regex = re.compile(r'^(https?|ftp):\/\/([A-Za-z0-9.-]+|\[[0-9a-fA-F:]+\])(:\d+)?(\/.*)?$', re.IGNORECASE)
    return re.match(regex, url) is not None

# --- Suspicious URL Check ---
def is_suspicious_url(url):
    if not isinstance(url, str) or not is_valid_url(url):
        return False
    suspicious_keywords = ['bit.ly', 'freestuff', 'fake', 'phishing', 'login', 'bank', 'secure']
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    return any(keyword in domain.lower() for keyword in suspicious_keywords)

# --- Unsubscribe Check ---
def check_unsubscribe(text):
    if not isinstance(text, str): return False
    text_lower = text.lower()
    unsubscribe_keywords = ['unsubscribe', 'click here to unsubscribe', 'opt-out', 'manage your preferences']
    return any(keyword in text_lower for keyword in unsubscribe_keywords)

# Load dataset
df = pd.read_csv("CEAS_08.csv", dtype=str, low_memory=False)
df = df.dropna(subset=["label"])

# Clean and prepare data
df["cleaned_body"] = df["body"].apply(clean_text)
df["cleaned_sender"] = df["sender"].apply(clean_text)
df["subject"] = df["subject"].fillna("")
df["combined_text"] = df["cleaned_sender"] + " " + df["subject"] + " " + df["cleaned_body"]
df["urls_in_body"] = df["body"].apply(extract_urls)
df["suspicious_urls"] = df["urls_in_body"].apply(lambda text: int(any(is_suspicious_url(url) for url in text.split())))
df["unsubscribe_present"] = df["body"].apply(lambda x: int(check_unsubscribe(x)))
df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

# Split data
X_train_text, X_test_text, y_train, y_test, train_suspicious, test_suspicious, train_unsub, test_unsub = train_test_split(
    df["combined_text"], df["label"], df["suspicious_urls"], df["unsubscribe_present"], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Combine features
X_train_extra = np.vstack((train_suspicious, train_unsub)).T
X_test_extra = np.vstack((test_suspicious, test_unsub)).T

X_train_final = hstack([X_train_tfidf, X_train_extra])
X_test_final = hstack([X_test_tfidf, X_test_extra])

# Model training
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train_final, y_train)

# Evaluate
cv_scores = cross_val_score(model, X_train_final, y_train, cv=5)
print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")

y_pred = model.predict(X_test_final)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
