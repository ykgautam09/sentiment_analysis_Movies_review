import pickle
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

with open('./app/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

with open('./app/model.pickle', 'rb') as f:
    model = pickle.load(f)


def clean_text(review):
    review = BeautifulSoup(review, 'lxml').text
    review = re.sub('[^a-zA-z]', ' ', review).lower()
    words = []
    for j in review.split():
        if j not in stopwords:
            words.append(stemmer.stem(j))
    return [' '.join(words)]


def predict_class(review):
    review = vectorizer.transform(clean_text(review))
    result = model.predict(review)[0]
    return result


if __name__ == "__main__":
    predict_class()
