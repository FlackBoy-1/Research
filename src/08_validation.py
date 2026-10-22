import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import os
# Определяем корневую папку проекта (на уровень выше src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# 1. Загружаем данные
df = pd.read_csv('corpus_preprocessed.csv', sep=';', encoding='utf-8-sig')

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# 2. Финальные параметры (наши k* и V*)
K_STAR = 24
MIN_DF = 3

# 3. Векторизация (V*)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=MIN_DF
)
X = vectorizer.fit_transform(df['lemmas'])
y = df['intent']

# 4. Разделение на Train/Test (80/20)
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
df_valid = df[df['intent'].isin(valid_classes)]

X_valid = vectorizer.fit_transform(df_valid['lemmas'])
y_valid = df_valid['intent']

# --- УМНАЯ СТРАТИФИКАЦИЯ ---
num_classes = len(y_valid.unique())
test_size_count = int(len(df_valid) * 0.2)

if test_size_count >= num_classes:
    # Если строк хватает, чтобы в тесте был хотя бы 1 представитель каждого класса
    stratify_param = y_valid
    print(f"Размер теста ({test_size_count}) >= классов ({num_classes}). Включаем стратификацию.")
else:
    # Если строк мало (пилот), отключаем
    stratify_param = None
    print(f"Размер теста ({test_size_count}) < классов ({num_classes}). Стратификация ОТКЛЮЧЕНА.")

X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, 
    test_size=0.2, 
    random_state=42, 
    stratify=stratify_param
)

# 5. Обучение SVD на Train (k*)
# Умная защита: если слов в Train меньше, чем K_STAR, берем столько осей, сколько слов
actual_k = min(K_STAR, X_train.shape[1])
if actual_k < K_STAR:
    print(f"ВНИМАНИЕ: В обучающей выборке всего {X_train.shape[1]} слов. Уменьшаем k с {K_STAR} до {actual_k}.")

svd = TruncatedSVD(n_components=actual_k, random_state=42)
X_train_lsa = svd.fit_transform(X_train)
X_test_lsa = svd.transform(X_test) # Трансформируем тест той же моделью!

# 6. Обучение классификатора
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_lsa, y_train)
y_pred = clf.predict(X_test_lsa)

# 6. Обучение классификатора
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_lsa, y_train)
y_pred = clf.predict(X_test_lsa)

# 7. Метрики
print("\n=== ОТЧЕТ ПО КЛАССАМ ===")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Матрица ошибок
# Берем только те метки, которые есть в y_test (чтобы матрица не была пустой)
labels_test = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_test, yticklabels=labels_test)
plt.title(f'Матрица ошибок (Test set)\nСтрок: {len(df)} | k*={K_STAR} | V*={MIN_DF}')
plt.ylabel('Истинный класс (Intent)')
plt.xlabel('Предсказанный класс (Intent)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Сохраняем
import os
from datetime import datetime
os.makedirs('plots/validation', exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'plots/validation/confusion_matrix_rows{len(df)}_{timestamp}.png'
plt.savefig(filename, dpi=300)
print(f"\nМатрица ошибок сохранена: {filename}")

plt.show()