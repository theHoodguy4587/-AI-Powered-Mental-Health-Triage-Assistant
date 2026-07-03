import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from preprocess import clean_text


val = pd.read_csv(
    "data/raw/val.csv"
)

val["cleaned_text"] = val["text"].apply(
    clean_text
)


model = joblib.load(
    "models/logistic_regression_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


X_val = tfidf.transform(
    val["cleaned_text"]
)


explainer = shap.LinearExplainer(
    model,
    X_val
)


# choose one sample
sample = X_val[0]


# calculate shap values
shap_values = explainer(sample)


# predicted class
prediction = model.predict(sample)[0]

print("Predicted class:", prediction)


# Find index of predicted class
class_index = list(model.classes_).index(prediction)


values = shap_values.values[0][:, class_index]


# Select SHAP values for predicted class



feature_names = tfidf.get_feature_names_out()


explanation_df = pd.DataFrame({

    "word": feature_names,

    "importance": values

})


explanation_df = explanation_df.sort_values(
    by="importance",
    key=abs,
    ascending=False
)


print(
    explanation_df.head(10)
)


# save plot

top_words = explanation_df.head(10)


plt.barh(
    top_words["word"],
    top_words["importance"]
)

plt.xlabel(
    "SHAP Importance"
)

plt.title(
    "Why the model predicted this class"
)

plt.gca().invert_yaxis()

plt.show()