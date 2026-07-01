from pathlib import Path

import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


    

train = pd.read_csv('data/raw/train.csv')
val = pd.read_csv('data/raw/val.csv')

train['cleaned_text'] = train['text'].apply(clean_text)
val['cleaned_text'] = val['text'].apply(clean_text)


tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2))

X_train = tfidf.fit_transform(train['cleaned_text'])
X_val = tfidf.transform(val['cleaned_text'])

y_train = train['label']
y_val = val['label']

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

joblib.dump(model, "models/logistic_regression_model.pkl")

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")



