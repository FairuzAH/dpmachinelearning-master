import pandas as pd
import re

from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load colloquial lexicon
normalizad_word = pd.read_csv("colloquial-indonesian-lexicon.csv")
normalizad_word_dict = dict(zip(normalizad_word.iloc[:, 0], normalizad_word.iloc[:, 1]))

# Preprocessing class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopwords = set(StopWordRemoverFactory().get_stop_words())

    def clean_text(self, text):
        text = text.lower()
        text = text.replace('\t', " ").replace('\n', " ").replace('\\', "")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", "", text).split())
        text = re.sub(r"http[s]?://", " ", text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
        text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def normalize(self, tokens):
        return [normalizad_word_dict.get(term, term) for term in tokens]

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stopwords]

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def transform(self, X, y=None):
        processed = []
        for text in X:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            normalized = self.normalize(tokens)
            no_stopwords = self.remove_stopwords(normalized)
            stemmed = self.stem(no_stopwords)
            processed.append(' '.join(stemmed))
        return processed

    def fit(self, X, y=None):
        return self
