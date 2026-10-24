import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_int(filename, default):
    try:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            return int(f.read().strip())
    except Exception:
        return default

K_STAR = load_int('best_k.txt', 51)
MIN_DF = load_int('best_min_df.txt', 6)

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

# БЕРЕМ ВООБЩЕ ВСЕ ДАННЫЕ
# Никаких разделений на train/test. Обучаем на всем, тестируем на всем.
# Мы даже увеличиваем max_iter, чтобы модель точно выучила все.
print(f"Всего строк в корпусе: {len(df)}")

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=MIN_DF
)

# Обучаем на всем тексте
X = vectorizer.fit_transform(df['lemmas'])
actual_k = min(K_STAR, X.shape[1] - 1)

svd = TruncatedSVD(n_components=actual_k, random_state=42)
X_lsa = svd.fit_transform(X)

# Обучаем классификатор на всех данных и сразу предсказываем на них же
clf = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
clf.fit(X_lsa, df['intent'])

y_pred = clf.predict(X_lsa)

f1 = f1_score(df['intent'], y_pred, average='macro', zero_division=0)

print("\n" + "="*50)
print("РЕЗУЛЬТАТ (OVERFITTING / TRAIN=TEST)")
print("="*50)
print(f"F1_intent (Macro): {f1:.4f}")
print(f"Accuracy: {np.mean(y_pred == df['intent']):.4f}")

print("\nДетальный отчет (полюбуйся):")
print(classification_report(df['intent'], y_pred, zero_division=0))