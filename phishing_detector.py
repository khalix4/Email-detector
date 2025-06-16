from flask import Flask, request, render_template
import pickle
import re
import string
import os
import numpy as np
from scipy.sparse import hstack
from train_model import extract_urls, is_suspicious_url, check_unsubscribe, clean_text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model_path = "phishing_model.pkl"
vectorizer_path = "vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, 'rb') as model_file, open(vectorizer_path, 'rb') as vec_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vec_file)
else:
    raise FileNotFoundError("Trained model or vectorizer file not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sender = request.form.get('sender', '')
    subject = request.form.get('subject', '')
    body = request.form.get('body', '')

    # URL extraction
    urls_in_body = extract_urls(body)
    urls_in_sender = extract_urls(sender)

    # Suspicious URL flag
    suspicious_urls_flag = int(any(is_suspicious_url(url) for url in urls_in_body.split()))

    # Unsubscribe feature
    has_unsubscribe = int(check_unsubscribe(body))

    # Combine features
    combined_text = f"{sender} {subject} {body} {urls_in_body} {urls_in_sender}"
    cleaned_input = clean_text(combined_text)
    tfidf_features = vectorizer.transform([cleaned_input])
    extra_features = np.array([[suspicious_urls_flag, has_unsubscribe]])
    features = hstack([tfidf_features, extra_features])

    # Predict
    prediction = model.predict(features)[0]
    result = "Phishing" if prediction == 1 else "Legitimate"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
