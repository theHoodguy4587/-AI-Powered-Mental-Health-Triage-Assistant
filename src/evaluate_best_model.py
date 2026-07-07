import pandas as pd
import joblib
import matplotlib.pyplot as plt

from preprocess import clean_text

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load model
best_model = joblib.load("models/best_model.pkl")

# Load validation data
val = pd.read_csv("data/raw/val.csv")

# Label mapping
risk_mapping = {
    "self.offmychest": "Low Risk",
    "self.Anxiety": "Moderate Risk",
    "self.depression": "Moderate Risk",
    "self.bipolar": "Moderate Risk",
    "self.SuicideWatch": "High Risk"
}

val["label"] = val["label"].map(risk_mapping)

# Clean text
val["text"] = val["text"].apply(clean_text)

# Features and labels
X_val = val["text"]
y_val = val["label"]

# Predictions
pred = best_model.predict(X_val)

# Report
print(classification_report(y_val, pred))

# Confusion Matrix
cm = confusion_matrix(
    y_val,
    pred,
    labels=best_model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - Best Model")

plt.tight_layout()

plt.savefig(
    "reports/best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()