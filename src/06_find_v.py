import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
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
df = df.dropna(subset=['intent'])

y_intent = df['intent']
num_rows = len(df)

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

try:
    with open(os.path.join(DATA_DIR, 'best_k.txt'), 'r') as f:
        K_STAR = int(f.read().strip())
    print(f"Автоматически загружен k* = {K_STAR} из файла.")
except Exception as e:
    print(f"[ОШИБКА] Файл best_k.txt не найден или поврежден ({e}). Запустите этап 5.")
    sys.exit(1)

plot_dir = os.path.join(PLOTS_DIR, 'v_analysis')
os.makedirs(plot_dir, exist_ok=True)

vocab_sizes = []
f1_scores = []
thresholds = []
cr_values = []

print(f"\nЗапуск поиска V* при фиксированном k*={K_STAR} (4 класса)...")

count_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    use_idf=False, norm=None, binary=False
)
count_matrix = count_vectorizer.fit_transform(df['lemmas'])
frequencies = count_matrix.sum(axis=0).A1
feature_names_all = count_vectorizer.get_feature_names_out()
freq_dict = dict(zip(feature_names_all, frequencies))

for current_min_df in range(1, 20):
    valid_features = []
    for word in feature_names_all:
        freq = freq_dict.get(word, 0)
        if freq >= current_min_df or word in keep_words:
            valid_features.append(word)
            
    vocab_size = len(valid_features)
    
    if vocab_size < K_STAR:
        print(f"min_df={current_min_df} (слов: {vocab_size}): Слишком мало слов для k*={K_STAR}, останавливаемся.")
        break

    thresholds.append(current_min_df)
    vocab_sizes.append(vocab_size)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        vec = TfidfVectorizer(
            ngram_range=(1, 2), 
            token_pattern=r'(?u)\b\w+\b',
            stop_words=final_stop_words,
            vocabulary=valid_features
        )
        pipeline = make_pipeline(vec, TruncatedSVD(n_components=K_STAR, random_state=42), clf)
        y_pred = cross_val_predict(pipeline, df['lemmas'], y_intent, cv=5)
        f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"[ОШИБКА] min_df={current_min_df} упал: {e}")
        f1 = np.nan
        
    f1_scores.append(f1)
    if not np.isnan(f1):
        print(f"min_df={current_min_df} (слов: {vocab_size}): F1 = {f1:.3f}")

total_vocab_size = vocab_sizes[0] 
cr_values = [total_vocab_size / v for v in vocab_sizes]

valid_indices = [i for i, f1 in enumerate(f1_scores) if not np.isnan(f1)]
if not valid_indices:
    print("[ОШИБКА] Все запуски упали. Невозможно выбрать V*.")
    sys.exit(1)

valid_f1 = [f1_scores[i] for i in valid_indices]
max_f1 = max(valid_f1)
tolerance = 0.005

candidate_indices = [i for i in valid_indices if f1_scores[i] >= (max_f1 - tolerance)]
best_idx = min(candidate_indices, key=lambda i: vocab_sizes[i])

best_min_df = thresholds[best_idx]
best_vocab_size = vocab_sizes[best_idx]
best_f1 = f1_scores[best_idx]
best_cr = cr_values[best_idx]

print(f"\n[+] Максимальный F1 (CV) = {max_f1:.3f}")
print(f"[+] Найдена оптимальная точка:")
print(f"    min_df = {best_min_df}")
print(f"    Размер словаря (V*) = {best_vocab_size}")
print(f"    Compression Ratio = {best_cr:.1f}x")
print(f"    F1 = {best_f1:.3f}")

with open(os.path.join(DATA_DIR, 'best_min_df.txt'), 'w') as f:
    f.write(str(best_min_df))
with open(os.path.join(DATA_DIR, 'best_v.txt'), 'w') as f:
    f.write(str(best_vocab_size))

# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.plot(vocab_sizes, f1_scores, color='tab:blue', marker='o', linewidth=2)
ax1.scatter([best_vocab_size], [best_f1], color='red', s=150, zorder=5, label=f'V* = {best_vocab_size} (min_df={best_min_df})')
ax1.set_xscale('log')
ax1.set_xlabel('Размер словаря (логарифмическая шкала)')
ax1.set_ylabel('Macro F1-Score (CV)')
ax1.set_title(f'F1 от размера словаря (4 класса)\nСтрок: {num_rows} | k* = {K_STAR}')
ax1.grid(True, alpha=0.3, which="both")
ax1.legend()

for i in range(len(thresholds)):
    if not np.isnan(f1_scores[i]):
        if thresholds[i] % 2 == 0 or thresholds[i] in [1, best_min_df]:
            ax1.annotate(f'min_df={thresholds[i]}', (vocab_sizes[i], f1_scores[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

ax2.plot(cr_values, f1_scores, color='tab:purple', marker='s', linewidth=2)
ax2.scatter([best_cr], [best_f1], color='red', s=150, zorder=5, label=f'V* = {best_vocab_size} (CR={best_cr:.1f}x)')
ax2.set_xlabel('Compression Ratio')
ax2.set_ylabel('Macro F1-Score (CV)')
ax2.set_title(f'Pareto-фронт: Цена сжатия')
ax2.grid(True, alpha=0.3)
ax2.legend()

for i in range(len(thresholds)):
    if not np.isnan(f1_scores[i]):
         if thresholds[i] % 2 == 0 or thresholds[i] in [1, best_min_df]:
            ax2.annotate(f'min_df={thresholds[i]}', (cr_values[i], f1_scores[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(plot_dir, f'pareto_vs_vocab_rows{num_rows}_k{K_STAR}_{timestamp}.png')
plt.savefig(filename, dpi=300)
plt.close()
print(f"\nГрафики сохранены в {filename}")