import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline

from text_processor import TextPreprocessor

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
conf_ktg = confusion_matrix(y_test_ktg, y_pred_ktg, labels=labels_ktg)
confusion_df_ktg = pd.DataFrame(conf_ktg, index=labels_ktg, columns=labels_ktg)

# Print accuracy and classification report
print("Accuracy (Kategori): {:.2f}%".format(accuracy_score(y_test_ktg, y_pred_ktg) * 100))
print("\nConfusion Matrix (Kategori):\n", confusion_df_ktg)
sns.heatmap(confusion_df_ktg, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Kategori")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report (Kategori):\n", classification_report(y_test_ktg, y_pred_ktg, target_names=labels_ktg))
