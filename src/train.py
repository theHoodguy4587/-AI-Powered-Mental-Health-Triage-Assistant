from pathlib import Path

import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


    

train = pd.read_csv('data/raw/train.csv')
val = pd.read_csv('data/raw/val.csv')

risk_mapping = {
    "self.offmychest": "Low Risk",
    "self.Anxiety": "Moderate Risk",
    "self.depression": "Moderate Risk",
    "self.bipolar": "Moderate Risk",
    "self.SuicideWatch": "High Risk"
}

train["risk_label"] = train["label"].map(risk_mapping)
val["risk_label"] = val["label"].map(risk_mapping)

train['cleaned_text'] = train['text'].apply(clean_text)
val['cleaned_text'] = val['text'].apply(clean_text)



tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=3,max_df=0.95)

X_train = tfidf.fit_transform(train['cleaned_text'])
X_val = tfidf.transform(val['cleaned_text'])

y_train = train['risk_label']
y_val = val['risk_label']

model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train,y_train)

joblib.dump(model, "models/logistic_regression_model.pkl")

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")



