import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
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

# ЧИТАЕМ k* ИЗ ФАЙЛА
try:
    with open(os.path.join(DATA_DIR, 'best_k.txt'), 'r') as f:
        K_STAR = int(f.read().strip())
    print(f"Автоматически загружен k* = {K_STAR} из файла.")
except:
    K_STAR = 24
    print("Файл best_k.txt не найден. Используем k*=24 по умолчанию.")

plot_dir = os.path.join(PLOTS_DIR, 'v_analysis')
os.makedirs(plot_dir, exist_ok=True)

vocab_sizes = []
f1_scores = []
thresholds = []
current_min_df = 1

print(f"Запуск поиска V* при фиксированном k*={K_STAR}...")

# ... дальше идет твой старый цикл while True, 
# но внутри него не забудь добавить class_weight='balanced':
# clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

while True:
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
        print(f"\nПри min_df={current_min_df} не осталось слов. Останавливаемся.")
        break
        
    if vocab_size < 2:
        print(f"\nПри min_df={current_min_df} осталось {vocab_size} слов. Это предел.")
        break

    thresholds.append(current_min_df)
    vocab_sizes.append(vocab_size)
    
    current_k = min(K_STAR, vocab_size)
    svd = TruncatedSVD(n_components=current_k, random_state=42)
    X_lsa = svd.fit_transform(X)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        y_pred = cross_val_predict(clf, X_lsa, y_intent, cv=5)
        f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    except:
        f1 = 0
        
    f1_scores.append(f1)
    
    print(f"min_df={current_min_df} (слов: {vocab_size}, осей: {current_k}): F1 = {f1:.3f}")
    current_min_df += 1

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Размер словаря (количество слов и биграмм)')
ax1.set_ylabel('Macro F1-Score', color=color)
ax1.plot(vocab_sizes, f1_scores, color=color, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

for i, txt in enumerate(thresholds):
    ax1.annotate(f'min_df={txt}', (vocab_sizes[i], f1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title(f'Зависимость F1 от размера словаря (V)\nСтрок: {num_rows} | k* = {K_STAR}', fontsize=14)
plt.grid(True, alpha=0.3)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = os.path.join(plot_dir, f'metrics_vs_vocab_rows{num_rows}_k{K_STAR}_{timestamp}.png')
plt.savefig(filename, dpi=300)
print(f"\nГрафик сохранен в {filename}")
#plt.show()