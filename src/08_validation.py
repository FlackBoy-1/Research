import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words(os.path.join(DATA_DIR, 'stop_words.txt'))
keep_words = load_words(os.path.join(DATA_DIR, 'keep_words.txt'))
final_stop_words = [word for word in stop_words if word not in keep_words]

# ЧИТАЕМ k* ИЗ ФАЙЛА
try:
    with open(os.path.join(DATA_DIR, 'best_k.txt'), 'r') as f:
        K_STAR = int(f.read().strip())
except:
    K_STAR = 24

MIN_DF = 3

# Разделение на Train/Test (ДО векторизации)
counts = df['intent'].value_counts()
valid_classes = counts[counts >= 2].index
df_valid = df[df['intent'].isin(valid_classes)]

test_size_count = int(len(df_valid) * 0.2)
num_classes = len(df_valid['intent'].unique())

if test_size_count >= num_classes:
    stratify_param = df_valid['intent']
else:
    stratify_param = None
    print("ВНИМАНИЕ: Стратификация отключена (мало данных).")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df_valid['lemmas'], df_valid['intent'], 
    test_size=0.2, random_state=42, stratify=stratify_param
)

# Векторизатор обучаем ТОЛЬКО на Train
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=MIN_DF
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# SVD
actual_k = min(K_STAR, X_train.shape[1])
svd = TruncatedSVD(n_components=actual_k, random_state=42)
X_train_lsa = svd.fit_transform(X_train)
X_test_lsa = svd.transform(X_test)

# Классификатор с балансировкой
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train_lsa, y_train)
y_pred = clf.predict(X_test_lsa)

print("\n=== ОТЧЕТ ПО КЛАССАМ ===")
print(classification_report(y_test, y_pred, zero_division=0))

labels_test = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_test, yticklabels=labels_test)
plt.title(f'Матрица ошибок (Test set)\nСтрок: {len(df)} | k*={K_STAR} | V*={MIN_DF}')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

val_dir = os.path.join(PLOTS_DIR, 'validation')
os.makedirs(val_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(val_dir, f'confusion_matrix_rows{len(df)}_{timestamp}.png')
plt.savefig(filename, dpi=300)
print(f"\nМатрица ошибок сохранена: {filename}")