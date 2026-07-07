import joblib
import numpy as np
from pathlib import Path
from src.preprocess import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)


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


def predict_text_details(text):
    """
    Predict class, confidence, and per-class probabilities for a given input text.
    """

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    probabilities = {}
    confidence = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(model.classes_, probs)
        }
        confidence = float(np.max(probs))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "cleaned_text": cleaned,
    }


# Example usage


if __name__ == "__main__":

    sample_text = "I feel hopeless and don't want to continue anymore"

    pred, conf = predict_text(sample_text)

    print("Input:", sample_text)
    print("Prediction:", pred)

    if conf is not None:
        print("Confidence:", round(conf * 100, 2), "%")
