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

# 1. Загружаем матрицу и данные
with open('matrix_x.pkl', 'rb') as f:
    X = pickle.load(f)
df = pd.read_csv('corpus_preprocessed.csv', sep=';', encoding='utf-8-sig')

# Целевые переменные
y_category = df['category']
y_intent = df['intent']
num_rows = len(df) # Считаем количество строк

# Создаем папку для графиков
os.makedirs('plots', exist_ok=True)

# Генерируем уникальное имя файла с датой, временем и количеством строк
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename_base = f"metrics_rows{num_rows}_k2-45_{timestamp}"

# 2. Задаем диапазон k
k_range = range(2, 46)
recon_errors = []
silhouettes = []
f1_scores = []

print(f"Запуск анализа для {num_rows} строк...")

for k in k_range:
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_lsa = svd.fit_transform(X)
    
    # 1. Reconstruction Error
    X_reconstructed = svd.inverse_transform(X_lsa)
    error = norm(X - X_reconstructed, 'fro')
    recon_errors.append(error)
    
    # 2. Silhouette Score
    try:
        sil = silhouette_score(X_lsa, y_category)
    except:
        sil = 0
    silhouettes.append(sil)
    
    # 3. Macro F1-Score
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_pred = cross_val_predict(clf, X_lsa, y_intent, cv=5)
    f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    f1_scores.append(f1)

# 3. Поиск локтей (алгоритмически через kneed)
# Ищем локти для ошибки (вогнутая кривая), силуэта и F1 (выпуклые)
knee_error = KneeLocator(k_range, recon_errors, curve='convex', direction='decreasing').elbow
knee_sil = KneeLocator(k_range, silhouettes, curve='concave', direction='increasing').elbow
knee_f1 = KneeLocator(k_range, f1_scores, curve='concave', direction='increasing').elbow

print(f"\n--- Результаты поиска локтей ---")
print(f"Локоть Reconstruction Error: k = {knee_error}")
print(f"Локоть Silhouette Score:     k = {knee_sil}")
print(f"Локоть F1-Score:             k = {knee_f1}")

# 4. Строим график
fig, ax1 = plt.subplots(figsize=(12, 7))

# Ось Y для ошибки (слева)
color = 'tab:red'
ax1.set_xlabel('Количество компонент (k)')
ax1.set_ylabel('Reconstruction Error', color=color)
ax1.plot(k_range, recon_errors, color=color, marker='o', label='Error')
ax1.tick_params(axis='y', labelcolor=color)
if knee_error:
    ax1.axvline(x=knee_error, color=color, linestyle=':', alpha=0.5)
    ax1.text(knee_error, max(recon_errors)*0.9, f' Elbow Error (k={knee_error})', color=color)

# Ось Y для силуэта и F1 (справа)
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Silhouette / F1 Score', color=color2)
ax2.plot(k_range, silhouettes, color='tab:green', marker='s', linestyle='--', label='Silhouette')
ax2.plot(k_range, f1_scores, color='tab:orange', marker='^', linestyle='-.', label='F1 Score')
ax2.tick_params(axis='y', labelcolor=color2)
if knee_f1:
    ax2.axvline(x=knee_f1, color='tab:orange', linestyle=':', alpha=0.5)
    ax2.text(knee_f1, max(f1_scores)*0.8, f' Elbow F1 (k={knee_f1})', color='tab:orange')

# Добавляем подпись с параметрами прямо на график
plt.title(f'Зависимость метрик от k\nСтрок в корпусе: {num_rows} | Диапазон k: 2-45', fontsize=14)
plt.grid(True, alpha=0.3)

# Объединяем легенды
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()

# Сохраняем график под уникальным именем
save_path = f'plots/{filename_base}.png'
plt.savefig(save_path, dpi=300)
print(f"\nГрафик сохранен в {save_path}")

plt.show()