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

# 1. Загружаем данные
df = pd.read_csv('corpus_preprocessed.csv', sep=';', encoding='utf-8-sig')
y_intent = df['intent']
num_rows = len(df)

# Загружаем наши стоп-слова
def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# Фиксируем k*
K_STAR = 24 

# Задаем пороги отсечения (min_df)
# min_df=1 означает "оставить все слова"
# min_df=2 означает "удалить слова, которые встречаются только 1 раз"
thresholds = [1, 2, 3, 4, 5, 7]
vocab_sizes = []
f1_scores = []

print(f"Запуск поиска V* при фиксированном k*={K_STAR}...")

for min_df in thresholds:
    # Векторизуем заново с новым порогом
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        token_pattern=r'(?u)\b\w+\b',
        stop_words=final_stop_words,
        min_df=min_df 
    )
    X = vectorizer.fit_transform(df['lemmas'])
    vocab_sizes.append(X.shape[1])
    
    # --- ЗАЩИТА ОТ ОШИБКИ ---
    # Если слов стало меньше, чем k*, берем количество слов
    current_k = min(K_STAR, X.shape[1])
    
    # Обучаем SVD на новом словаре
    svd = TruncatedSVD(n_components=current_k, random_state=42)
    X_lsa = svd.fit_transform(X)
    
    # Считаем F1
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_pred = cross_val_predict(clf, X_lsa, y_intent, cv=5)
    f1 = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    f1_scores.append(f1)
    
    print(f"min_df={min_df} (слов в словаре: {X.shape[1]}, осей: {current_k}): F1 = {f1:.3f}")

# 2. Строим график
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Размер словаря (количество слов и биграмм)')
ax1.set_ylabel('Macro F1-Score', color=color)
ax1.plot(vocab_sizes, f1_scores, color=color, marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Добавляем подписи точек (порог min_df)
for i, txt in enumerate(thresholds):
    ax1.annotate(f'min_df={txt}', (vocab_sizes[i], f1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title(f'Зависимость F1 от размера словаря (V)\nСтрок: {num_rows} | Фиксированный k* = {K_STAR}', fontsize=14)
plt.grid(True, alpha=0.3)

# Сохраняем уникальный график
os.makedirs('plots', exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'plots/metrics_vs_vocab_rows{num_rows}_k{K_STAR}_{timestamp}.png'
plt.savefig(filename, dpi=300)
print(f"\nГрафик сохранен: {filename}")

plt.show()