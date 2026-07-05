import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from preprocess import clean_text


# Load validation data
val = pd.read_csv(
    "data/raw/val.csv"
)

risk_mapping = {
    "self.offmychest": "Low Risk",
    "self.Anxiety": "Moderate Risk",
    "self.depression": "Moderate Risk",
    "self.bipolar": "Moderate Risk",
    "self.SuicideWatch": "High Risk"
}

val['risk_label'] = val['label'].map(risk_mapping)

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
y_val = val['risk_label']


# Prediction
pred = model.predict(X_val)

cm = confusion_matrix(y_val,pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

print(classification_report(
    y_val,
    pred
))


print(
    "Accuracy:",
    accuracy_score(y_val,pred)
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()