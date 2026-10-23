import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from kneed import KneeLocator
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')
y_intent = df['intent']
num_rows = len(df)

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words(os.path.join(DATA_DIR, 'stop_words.txt'))
keep_words = load_words(os.path.join(DATA_DIR, 'keep_words.txt'))
final_stop_words = [word for word in stop_words if word not in keep_words]

try:
    with open(os.path.join(DATA_DIR, 'best_k.txt'), 'r') as f:
        K_STAR = int(f.read().strip())
    print(f"Автоматически загружен k* = {K_STAR} из файла.")
except:
    K_STAR = 67

plot_dir = os.path.join(PLOTS_DIR, 'v_analysis')
os.makedirs(plot_dir, exist_ok=True)

vocab_sizes = []
f1_scores = []
thresholds = []

print(f"\nЗапуск поиска V* при фиксированном k*={K_STAR}...")

# Расширяем перебор до 10, чтобы точно увидеть падение
for current_min_df in range(1, 11):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        token_pattern=r'(?u)\b\w+\b',
        stop_words=final_stop_words,
        min_df=current_min_df 
    )
    
    try:
        X = vectorizer.fit_transform(df['lemmas'])
        vocab_size = X.shape[1]
    except ValueError:
        continue
        
    if vocab_size < K_STAR:
        print(f"min_df={current_min_df} (слов: {vocab_size}): Слишком мало слов для k*={K_STAR}, останавливаемся.")
        break

    thresholds.append(current_min_df)
    vocab_sizes.append(vocab_size)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1,2), token_pattern=r'(?u)\b\w+\b', stop_words=final_stop_words, min_df=current_min_df),
                                 TruncatedSVD(n_components=K_STAR, random_state=42),
                                 clf)
        y_pred = cross_val_predict(pipeline, df['lemmas'], y_intent, cv=5)
        f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    except Exception as e:
        f1 = 0
        
    f1_scores.append(f1)
    print(f"min_df={current_min_df} (слов: {vocab_size}): F1 = {f1:.3f}")

# --- УМНЫЙ ВЫБОР V* (Компромисс между F1 и сжатием) ---
max_f1 = max(f1_scores)
tolerance = 0.01 # Допустимая потеря качества (1%)

# Находим точки, где F1 не упал больше, чем на 1%
valid_indices = [i for i, f1 in enumerate(f1_scores) if f1 >= (max_f1 - tolerance)]

# Из этих точек выбираем ту, где словарь МИНИМАЛЕН (то есть максимальный min_df)
best_idx = valid_indices[-1]

best_min_df = thresholds[best_idx]
best_vocab_size = vocab_sizes[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n[+] Максимальный F1 = {max_f1:.3f} (при min_df={thresholds[np.argmax(f1_scores)]})")
print(f"[+] Допустимая просадка (tolerance) = {tolerance}")
print(f"[+] Найдена оптимальная точка (макс. сжатие при потере F1 не более 1%):")
print(f"    min_df = {best_min_df}")
print(f"    Размер словаря (V*) = {best_vocab_size}")
print(f"    F1 = {best_f1:.3f}")

# Сохраняем
with open(os.path.join(DATA_DIR, 'best_min_df.txt'), 'w') as f:
    f.write(str(best_min_df))
with open(os.path.join(DATA_DIR, 'best_v.txt'), 'w') as f:
    f.write(str(best_vocab_size))

# Построение графика
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Размер словаря (количество слов и биграмм)')
ax1.set_ylabel('Macro F1-Score', color=color)
ax1.plot(vocab_sizes, f1_scores, color=color, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# ВОТ ЗДЕСЬ ставим точку ровно туда, где best_idx
ax1.scatter([best_vocab_size], [best_f1], color='red', s=100, zorder=5, label=f'V* = {best_vocab_size} (min_df={best_min_df})')
ax1.legend()

# Подписываем точки
for i in range(len(thresholds)):
    ax1.annotate(f'min_df={thresholds[i]}', (vocab_sizes[i], f1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title(f'Зависимость F1 от размера словаря (V)\nСтрок: {num_rows} | k* = {K_STAR}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(plot_dir, f'metrics_vs_vocab_rows{num_rows}_k{K_STAR}_{timestamp}.png')
plt.savefig(filename, dpi=300)
print(f"\nГрафик сохранен в {filename}")