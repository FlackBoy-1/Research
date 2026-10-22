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

# ГИБКИЙ ДИАПАЗОН K (до 100 или до количества строк)
max_possible_k = min(X.shape[1], len(df) - 1, 100)
k_range = range(2, max_possible_k + 1)

recon_errors = []
silhouettes = []
f1_scores = []

print(f"Пересчет метрик для анализа (k от 2 до {max_possible_k})...")
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
    
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') # Балансировка
    y_pred = cross_val_predict(clf, X_lsa, y_intent, cv=5)
    f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    f1_scores.append(f1)

max_f1_idx = np.argmax(f1_scores)
k_star_candidate = k_range[max_f1_idx]
max_f1_val = f1_scores[max_f1_idx]

min_error = min(recon_errors)
min_error_idx = recon_errors.index(min_error)
candidate_error = recon_errors[max_f1_idx]
error_increase_pct = ((candidate_error - min_error) / min_error) * 100
candidate_sil = silhouettes[max_f1_idx]

print(f"\nЛучший F1 достигается при k = {k_star_candidate} (F1 = {max_f1_val:.3f})")

# Автоматически сохраняем k* для следующих скриптов
with open(os.path.join(DATA_DIR, 'best_k.txt'), 'w') as f:
    f.write(str(k_star_candidate))
print(f"k* = {k_star_candidate} сохранен в data/best_k.txt")