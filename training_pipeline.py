import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from text_processor import TextPreprocessor


start_time = time.time()

# Load dataset
df = pd.read_excel("annotated_all_agreed.xlsx")

# --------------------- MODEL 1: Relevansi Classification ---------------------
print("Training Model 1: Relevansi Classification")
X_relevansi = df['tweet']
y_relevansi = df['relevansi']

x_train_rel, x_test_rel, y_train_rel, y_test_rel = train_test_split(X_relevansi, y_relevansi, test_size=0.20, random_state=0)

pipeline_relevansi = ImbPipeline([
    ('preprocessing', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 1))),
    ('classifier', LinearSVC())
])

pipeline_relevansi.fit(x_train_rel, y_train_rel)
joblib.dump(pipeline_relevansi, 'model_relevansi_SVM_full_pipeline.pkl')

y_pred_rel = pipeline_relevansi.predict(x_test_rel)

# Display metrics
accuracy_rel = accuracy_score(y_test_rel, y_pred_rel)
f1_rel = classification_report(y_test_rel, y_pred_rel, output_dict=True)['accuracy']
print(f"Accuracy (Relevansi): {accuracy_rel * 100:.2f}%")
print(f"F1 Score (Relevansi): {f1_rel:.2f}")

# --------------------- MODEL 2: Kategori Classification ---------------------
print("Training Model 2: Kategori Classification")

# Exclude rows where 'kategori' is "Tidak"
df_filtered = df[df['kategori'] != 'Tidak']

# Now split the data again
X_kategori = df_filtered['tweet']
y_kategori = df_filtered['kategori']

# Train-test split
x_train_ktg, x_test_ktg, y_train_ktg, y_test_ktg = train_test_split(X_kategori, y_kategori, test_size=0.20, random_state=0)

# Update labels_ktg (without "Tidak")
labels_ktg = ['Terindikasi', 'Selfdiagnosed', 'Penderita', 'Penyintas']

# Create and fit the pipeline
pipeline_kategori = ImbPipeline([
    ('preprocessing', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 1))),
    ('classifier', RandomForestClassifier())
])

pipeline_kategori.fit(x_train_ktg, y_train_ktg)
joblib.dump(pipeline_kategori, 'model_kategori_RF_full_pipeline.pkl')

# Evaluate model
y_pred_ktg = pipeline_kategori.predict(x_test_ktg)

# Display metrics
accuracy_ktg = accuracy_score(y_test_ktg, y_pred_ktg)
f1_ktg = classification_report(y_test_ktg, y_pred_ktg, output_dict=True)['accuracy']
print(f"Accuracy (Kategori): {accuracy_ktg * 100:.2f}%")
print(f"F1 Score (Kategori): {f1_ktg:.2f}")

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")