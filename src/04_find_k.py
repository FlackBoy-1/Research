import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from scipy.linalg import norm
from kneed import KneeLocator
import os
from datetime import datetime
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig', keep_default_na=False)

# --- МАППИНГ 16 КЛАССОВ В 4 МАКРОКАТЕГОРИИ ---
MACRO_MAPPING = {
    'военная_угроза': 'угроза_военная',
    'техногенная_угроза': 'угроза_техногенная',
    'травма': 'угроза_общая', 'погибший': 'угроза_общая',
    'природная_угроза': 'угроза_общая', 'животная_угроза': 'угроза_общая',
    'заблудился': 'угроза_общая', 'потерялся': 'угроза_общая',
    'нет_припасов': 'угроза_общая', 'техническая_неисправность': 'угроза_общая', 
    'погода': 'не_угроза', 'позиция': 'не_угроза',
    'статус_группы': 'не_угроза', 'проверка_связи': 'не_угроза',
    'запрос_информации': 'не_угроза', 'запрос_маршрута': 'не_угроза'
}
df['intent'] = df['intent'].map(MACRO_MAPPING)
df = df.dropna(subset=['intent']) # Удаляем строки, если вдруг попались неизвестные классы

y_category = df['intent'] # Силуэт считаем по 4 классам
y_intent = df['intent']   # F1 считаем по 4 классам
num_rows = len(df)

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# Восстанавливаем матрицу X
model_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
matrix_path = os.path.join(DATA_DIR, 'matrix_x.pkl')

if not os.path.exists(model_path) or not os.path.exists(matrix_path):
    print("[ОШИБКА] Файлы vectorizer.pkl или matrix_x.pkl не найдены. Запустите этап 3.")
    sys.exit(1)

with open(model_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(matrix_path, 'rb') as f:
    X = pickle.load(f)

# Защита от несовпадения размеров, если корпус менялся
if X.shape[0] != num_rows:
    print("[ВНИМАНИЕ] Размер матрицы X не совпадает с размером датасета. Рекомендуется перезапустить 03_lsa.py. Продолжаем...")

num_features = X.shape[1]
plot_dir = os.path.join(PLOTS_DIR, 'k_analysis')
os.makedirs(plot_dir, exist_ok=True)

max_possible_k = min(num_features, num_rows - 1, 150) 
k_range = range(2, max_possible_k + 1)

recon_errors = []
silhouettes = []
f1_scores = []

print(f"Запуск анализа для {num_rows} строк (4 макрокласса). Диапазон k: 2 - {max_possible_k}")

for k in k_range:
    # 1. Reconstruction Error
    svd = TruncatedSVD(n_components=k, random_state=42)
    X_lsa = svd.fit_transform(X)
    X_reconstructed = svd.inverse_transform(X_lsa)
    error = norm(X - X_reconstructed, 'fro')
    recon_errors.append(error)
    
    # 2. Silhouette
    try:
        if len(np.unique(y_category)) > 1:
            sil = silhouette_score(X_lsa, y_category)
        else:
            sil = np.nan
    except Exception as e:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Ошибка при расчете Silhouette на k={k}: {e}")
        sil = np.nan
    silhouettes.append(sil)
    
    # 3. F1-Score (без утечки)
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        pipeline = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2), 
                token_pattern=r'(?u)\b\w+\b',
                stop_words=final_stop_words,
                min_df=1
            ),
            TruncatedSVD(n_components=k, random_state=42), 
            clf
        )
        y_pred = cross_val_predict(pipeline, df['lemmas'], y_intent, cv=5, n_jobs=2)
        f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Ошибка при CV на k={k}: {e}")
        f1 = np.nan
        
    f1_scores.append(f1)

metrics_df = pd.DataFrame({
    'k': list(k_range),
    'recon_error': recon_errors,
    'silhouette': silhouettes,
    'f1_score': f1_scores
})
metrics_df.to_csv(os.path.join(DATA_DIR, 'k_metrics.csv'), sep=';', index=False)

valid_mask = ~np.isnan(f1_scores)
k_valid = np.array(list(k_range))[valid_mask]
f1_valid = np.array(f1_scores)[valid_mask]

if len(k_valid) > 0:
    knee_error = KneeLocator(k_range, recon_errors, curve='convex', direction='decreasing').elbow
    knee_sil = KneeLocator(k_range, silhouettes, curve='concave', direction='increasing').elbow
    knee_f1 = KneeLocator(k_valid, f1_valid, curve='concave', direction='increasing').elbow
else:
    knee_error, knee_sil, knee_f1 = None, None, None

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
    ax2.axvline(x=knee_f1, color='tab:orange', linestyle=':', alpha=0.5, label='Локоть F1')

plt.title(f'Зависимость метрик от k (4 класса)\nСтрок: {num_rows} | k: 2-{max_possible_k}', fontsize=14)
plt.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.tight_layout()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(plot_dir, f'metrics_rows{num_rows}_k2-{max_possible_k}_{timestamp}.png')
plt.savefig(filename, dpi=300)
plt.close()
print(f"\nГрафик сохранен в {filename}")