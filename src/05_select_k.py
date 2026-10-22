import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy.linalg import norm
import os

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

with open(os.path.join(DATA_DIR, 'matrix_x.pkl'), 'rb') as f:
    X = pickle.load(f)
df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')
y_category = df['category']
y_intent = df['intent']

k_range = range(2, 46)
recon_errors = []
silhouettes = []
f1_scores = []

print("Пересчет метрик для анализа...")
for k in k_range:
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_lsa = svd.fit_transform(X)
    
    error = norm(X - svd.inverse_transform(X_lsa), 'fro')
    recon_errors.append(error)
    
    try:
        sil = silhouette_score(X_lsa, y_category)
    except:
        sil = 0
    silhouettes.append(sil)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_pred = cross_val_predict(clf, X_lsa, y_intent, cv=5)
    f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    f1_scores.append(f1)

max_f1_idx = np.argmax(f1_scores)
k_star_candidate = k_range[max_f1_idx]
max_f1_val = f1_scores[max_f1_idx]

print(f"\nЛучший F1 достигается при k = {k_star_candidate} (F1 = {max_f1_val:.3f})")

min_error = min(recon_errors)
min_error_idx = recon_errors.index(min_error)
candidate_error = recon_errors[max_f1_idx]
error_increase_pct = ((candidate_error - min_error) / min_error) * 100
candidate_sil = silhouettes[max_f1_idx]

print(f"Проверка условия (ошибка не более +5%):")
print(f"  Минимальная ошибка (при k={k_range[min_error_idx]}): {min_error:.2f}")
print(f"  Ошибка при k={k_star_candidate}: {candidate_error:.2f}")
print(f"  Увеличение: {error_increase_pct:.2f}%")

if error_increase_pct <= 5.0:
    print("  ✅ Условие по ошибке выполнено.")
else:
    print("  ❌ Условие по ошибке НЕ выполнено (ошибка выросла слишком сильно).")

print(f"\nПроверка силуэта (должен быть >= 0.1):")
print(f"  Силуэт при k={k_star_candidate}: {candidate_sil:.3f}")
if candidate_sil >= 0.1:
    print("  ✅ Условие по силуэту выполнено.")
else:
    print("  ❌ Условие по силуэту НЕ выполнено (классы перемешаны).")

print("\n--- ИТОГОВЫЙ ВЕРДИКТ ---")
if candidate_sil >= 0.1 and error_increase_pct <= 5.0:
    print(f"Выбран k* = {k_star_candidate} (прошел все проверки).")
else:
    print(f"Проверки не пройдены (данных слишком мало).")
    print(f"Согласно правилу из journal.md: 'В крайнем случае — выбор по максимуму F1'.")
    print(f"Принимаем k* = {k_star_candidate} с пометкой о малом объеме данных.")