import re
import nltk

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+','',text)

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)