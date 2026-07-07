import pandas as pd
import joblib
from preprocess import clean_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# =========================
# Load data
# =========================

train = pd.read_csv("data/raw/train.csv")
val = pd.read_csv("data/raw/val.csv")


# =========================
# Map labels (IMPORTANT)
# =========================

risk_mapping = {
    "self.offmychest": "Low Risk",
    "self.Anxiety": "Moderate Risk",
    "self.depression": "Moderate Risk",
    "self.bipolar": "Moderate Risk",
    "self.SuicideWatch": "High Risk"
}

train["label"] = train["label"].map(risk_mapping)
val["label"] = val["label"].map(risk_mapping)


# =========================
# Clean text
# =========================

train["text"] = train["text"].apply(clean_text)
val["text"] = val["text"].apply(clean_text)


X_train = train["text"]
y_train = train["label"]

X_val = val["text"]
y_val = val["label"]


# =========================
# Pipeline
# =========================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000))
])


# =========================
# Hyperparameters grid
# =========================

param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [2, 5],
    "tfidf__max_df": [0.9, 0.95],

    "model__C": [0.1, 1, 5, 10],
    "model__class_weight": [None, "balanced"]
}


# =========================
# Grid Search
# =========================

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1_weighted",
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)


# =========================
# Best model
# =========================

print("Best parameters:\n", grid.best_params_)

best_model = grid.best_estimator_


# =========================
# Evaluate on validation set
# =========================

pred = best_model.predict(X_val)

print("\nClassification Report:\n")
print(classification_report(y_val, pred))




joblib.dump(best_model, "models/best_model.pkl")

print("\nModel saved successfully!")