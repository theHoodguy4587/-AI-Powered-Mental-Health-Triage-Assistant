import pandas as pd
import joblib

from sklearn.metrics import classification_report, accuracy_score

from preprocess import clean_text


# Load validation data
val = pd.read_csv(
    "data/raw/val.csv"
)


# Create clean_text column
val['clean_text'] = val['text'].apply(clean_text)


# Load saved objects
model = joblib.load(
    "models/logistic_regression_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# Transform validation text
X_val = tfidf.transform(
    val['clean_text']
)


# Labels
y_val = val['label']


# Prediction
pred = model.predict(X_val)


print(classification_report(
    y_val,
    pred
))


print(
    "Accuracy:",
    accuracy_score(y_val,pred)
)