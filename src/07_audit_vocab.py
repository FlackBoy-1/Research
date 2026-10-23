import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# ЧИТАЕМ min_df* ИЗ ФАЙЛА
try:
    with open(os.path.join(DATA_DIR, 'best_min_df.txt'), 'r') as f:
        BEST_MIN_DF = int(f.read().strip())
    print(f"Автоматически загружен min_df* = {BEST_MIN_DF} из файла.")
except:
    BEST_MIN_DF = 3
    print("Файл best_min_df.txt не найден. Используем min_df=3 по умолчанию.")

# 1. Получаем словарь ПОСЛЕ применения min_df (то, что реально используется в V*)
vectorizer_kept = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=BEST_MIN_DF 
)
X_kept = vectorizer_kept.fit_transform(df['lemmas'])
feature_names_kept = vectorizer_kept.get_feature_names_out()

# 2. Получаем ПОЛНЫЙ словарь (до применения min_df, то есть min_df=1)
vectorizer_all = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=1 
)
X_all = vectorizer_all.fit_transform(df['lemmas'])
feature_names_all = vectorizer_all.get_feature_names_out()

# Считаем частоту каждого слова/биграммы для наглядности
frequencies = X_all.sum(axis=0).A1
freq_dict = dict(zip(feature_names_all, frequencies))

# 3. Сохраняем ОСТАВШИЕСЯ слова
kept_path = os.path.join(DATA_DIR, 'vocab_kept.txt')
with open(kept_path, 'w', encoding='utf-8') as f:
    f.write(f"# Словарь V* (оставшиеся слова, min_df={BEST_MIN_DF})\n")
    f.write(f"# Всего слов: {len(feature_names_kept)}\n\n")
    for word in feature_names_kept:
        f.write(f"{word}\n")

# 4. Сохраняем УДАЛЕННЫЕ слова (с указанием частоты)
removed_path = os.path.join(DATA_DIR, 'vocab_removed.txt')
removed_words = [w for w in feature_names_all if w not in feature_names_kept]

with open(removed_path, 'w', encoding='utf-8') as f:
    f.write(f"# Удаленные слова (встречались реже, чем {BEST_MIN_DF} раз)\n")
    f.write(f"# Всего слов: {len(removed_words)}\n\n")
    # Сортируем по убыванию частоты (самые частые из удаленных — вверху)
    removed_sorted = sorted(removed_words, key=lambda w: freq_dict[w], reverse=True)
    for word in removed_sorted:
        f.write(f"{word} (частота: {int(freq_dict[word])})\n")

print("\n=== АУДИТ СЛОВАРЯ ЗАВЕРШЕН ===")
print(f"Всего слов до фильтрации (min_df=1): {len(feature_names_all)}")
print(f"Оставлено в V* (min_df={BEST_MIN_DF}): {len(feature_names_kept)}")
print(f"Удалено как редкие: {len(removed_words)}")
print(f"\n[+] Оставшиеся слова сохранены в: {kept_path}")
print(f"[+] Удаленные слова (с частотой) сохранены в: {removed_path}")

print("\n=== ПРОВЕРКА КРИТИЧЕСКИХ СЛОВ В V* ===")
critical_words = ['лавина', 'медведь', 'обморожение', 'кровотечение', 'травма', 'помочь', 'спасатель']
for word in critical_words:
    if word in feature_names_kept:
        print(f"[+] {word} — осталось в словаре.")
    else:
        print(f"[-] {word} — ПОТЕРЯНО! (частота: {int(freq_dict.get(word, 0))})")