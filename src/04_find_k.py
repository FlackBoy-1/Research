import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy.linalg import norm
from kneed import KneeLocator
import os
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

with open(os.path.join(DATA_DIR, 'matrix_x.pkl'), 'rb') as f:
    X = pickle.load(f)
df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

y_category = df['category']
y_intent = df['intent']
num_rows = len(df)
num_features = X.shape[1]

plot_dir = os.path.join(PLOTS_DIR, 'k_analysis')
os.makedirs(plot_dir, exist_ok=True)

max_possible_k = min(num_features, num_rows - 1, 100) 
k_range = range(2, max_possible_k + 1)

recon_errors = []
silhouettes = []
f1_scores = []

print(f"Запуск анализа для {num_rows} строк. Диапазон k: 2 - {max_possible_k}")

for k in k_range:
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_lsa = svd.fit_transform(X)
    
    X_reconstructed = svd.inverse_transform(X_lsa)
    error = norm(X - X_reconstructed, 'fro')
    recon_errors.append(error)
    
    try:
        sil = silhouette_score(X_lsa, y_category)
    except:
        sil = 0
    silhouettes.append(sil)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        y_pred = cross_val_predict(clf, X_lsa, y_intent, cv=5)
        f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    except:
        f1 = 0
    f1_scores.append(f1)

knee_error = KneeLocator(k_range, recon_errors, curve='convex', direction='decreasing').elbow
knee_sil = KneeLocator(k_range, silhouettes, curve='concave', direction='increasing').elbow
knee_f1 = KneeLocator(k_range, f1_scores, curve='concave', direction='increasing').elbow

print(f"\n--- Результаты поиска локтей ---")
print(f"Локоть Reconstruction Error: k = {knee_error}")
print(f"Локоть Silhouette Score:     k = {knee_sil}")
print(f"Локоть F1-Score:             k = {knee_f1}")

fig, ax1 = plt.subplots(figsize=(12, 7))
color = 'tab:red'
ax1.set_xlabel('Количество компонент (k)')
ax1.set_ylabel('Reconstruction Error', color=color)
ax1.plot(k_range, recon_errors, color=color, marker='o', label='Error')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Silhouette / F1 Score', color=color2)
ax2.plot(k_range, silhouettes, color='tab:green', marker='s', linestyle='--', label='Silhouette')
ax2.plot(k_range, f1_scores, color='tab:orange', marker='^', linestyle='-.', label='F1 Score')
ax2.tick_params(axis='y', labelcolor=color2)

if knee_f1:
    ax2.axvline(x=knee_f1, color='tab:orange', linestyle=':', alpha=0.5)

plt.title(f'Зависимость метрик от k\nСтрок: {num_rows} | k: 2-{max_possible_k}', fontsize=14)
plt.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.tight_layout()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(plot_dir, f'metrics_rows{num_rows}_k2-{max_possible_k}_{timestamp}.png')
plt.savefig(filename, dpi=300)
print(f"\nГрафик сохранен в {filename}")
#plt.show()