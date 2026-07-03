import joblib
import numpy as np
from preprocess import clean_text



# Load trained artifacts


model = joblib.load("models/logistic_regression_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Prediction function


def predict_text(text):
    """
    Predict class and confidence for a given input text.
    """

    # 1. Clean input
    cleaned = clean_text(text)

    # 2. Transform using TF-IDF
    vector = tfidf.transform([cleaned])

    # 3. Predict class
    prediction = model.predict(vector)[0]

    # 4. Get probability scores (if supported)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        confidence = np.max(probs)
    else:
        confidence = None

    return prediction, confidence


# Example usage


if __name__ == "__main__":

    sample_text = "I feel hopeless and don't want to continue anymore"

    pred, conf = predict_text(sample_text)

    print("Input:", sample_text)
    print("Prediction:", pred)

    if conf is not None:
        print("Confidence:", round(conf * 100, 2), "%")