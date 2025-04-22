import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


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

# Load annotated dataset
df = pd.read_excel("annotated_all_agreed.xlsx")

# --------------------- MODEL 1: Relevansi Classification ---------------------
print("Training Model 1: Relevansi Classification")
X_relevansi = df['tweet']
y_relevansi = df['relevansi']

x_train_rel, x_test_rel, y_train_rel, y_test_rel = train_test_split(X_relevansi, y_relevansi, test_size=0.20, random_state=0)

pipeline_relevansi = ImbPipeline([
    ('preprocessing', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 1))),
    #    ('smote', SMOTE(random_state=12)),
    ('classifier', LinearSVC())
])

pipeline_relevansi.fit(x_train_rel, y_train_rel)
joblib.dump(pipeline_relevansi, 'model_relevansi_SVM_full_pipeline.pkl')

y_pred_rel = pipeline_relevansi.predict(x_test_rel)
labels_rel = ['Ya', 'Tidak']
conf_rel = confusion_matrix(y_test_rel, y_pred_rel, labels=labels_rel)
confusion_df_rel = pd.DataFrame(conf_rel, index=labels_rel, columns=labels_rel)

print("Accuracy (Relevansi): {:.2f}%".format(accuracy_score(y_test_rel, y_pred_rel) * 100))
print("\nConfusion Matrix (Relevansi):\n", confusion_df_rel)
sns.heatmap(confusion_df_rel, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Relevansi")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report (Relevansi):\n", classification_report(y_test_rel, y_pred_rel, target_names=labels_rel))


# --------------------- MODEL 2: Kategori Classification ---------------------
print("Training Model 2: Kategori Classification")
X_kategori = df['tweet']
y_kategori = df['kategori']

x_train_ktg, x_test_ktg, y_train_ktg, y_test_ktg = train_test_split(X_kategori, y_kategori, test_size=0.20, random_state=0)

pipeline_kategori = ImbPipeline([
    ('preprocessing', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 1))),
    #    ('smote', SMOTE(random_state=12)),
    ('classifier', RandomForestClassifier())
])

pipeline_kategori.fit(x_train_ktg, y_train_ktg)
joblib.dump(pipeline_kategori, 'model_kategori_RF_full_pipeline.pkl')

y_pred_ktg = pipeline_kategori.predict(x_test_ktg)
labels_ktg = ['Terindikasi', 'Selfdiagnosed', 'Penderita', 'Penyintas']
conf_ktg = confusion_matrix(y_test_ktg, y_pred_ktg, labels=labels_ktg)
confusion_df_ktg = pd.DataFrame(conf_ktg, index=labels_ktg, columns=labels_ktg)

print("Accuracy (Kategori): {:.2f}%".format(accuracy_score(y_test_ktg, y_pred_ktg) * 100))
print("\nConfusion Matrix (Kategori):\n", confusion_df_ktg)
sns.heatmap(confusion_df_ktg, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Kategori")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report (Kategori):\n", classification_report(y_test_ktg, y_pred_ktg, target_names=labels_ktg))
