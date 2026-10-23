import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

try:
    with open(os.path.join(DATA_DIR, 'best_min_df.txt'), 'r') as f:
        BEST_MIN_DF = int(f.read().strip())
    print(f"Автоматически загружен min_df* = {BEST_MIN_DF} из файла.")
except Exception as e:
    print(f"[ОШИБКА] Файл best_min_df.txt не найден или поврежден ({e}). Запустите этап 6.")
    sys.exit(1)

# 1. Считаем частоты всех слов (униграммы и биграммы)
# Используем CountVectorizer (через Tfidf с use_idf=False) для точных частот
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

# 2. Формируем итоговый словарь V*
feature_names_kept = []
removed_words = []

for word in feature_names_all:
    # Защита от KeyError (на всякий случай, хотя тут все слова из корпуса)
    freq = freq_dict.get(word, 0)
    if freq >= BEST_MIN_DF or word in keep_words:
        feature_names_kept.append(word)
    else:
        removed_words.append(word)

# 3. Сохраняем ОСТАВШИЕСЯ слова
kept_path = os.path.join(DATA_DIR, 'vocab_kept.txt')
with open(kept_path, 'w', encoding='utf-8') as f:
    f.write(f"# Словарь V* (оставшиеся слова, min_df={BEST_MIN_DF})\n")
    f.write(f"# Всего слов: {len(feature_names_kept)}\n\n")
    for word in feature_names_kept:
        f.write(f"{word}\n")

# 4. Сохраняем УДАЛЕННЫЕ слова
removed_path = os.path.join(DATA_DIR, 'vocab_removed.txt')
with open(removed_path, 'w', encoding='utf-8') as f:
    f.write(f"# Удаленные слова (встречались реже, чем {BEST_MIN_DF} раз)\n")
    f.write(f"# Всего слов: {len(removed_words)}\n\n")
    removed_sorted = sorted(removed_words, key=lambda w: freq_dict[w], reverse=True)
    for word in removed_sorted:
        f.write(f"{word} (частота: {int(freq_dict[word])})\n")

print("\n=== АУДИТ СЛОВАРЯ ЗАВЕРШЕН ===")
print(f"Всего фич до фильтрации (min_df=1): {len(feature_names_all)}")
print(f"Оставлено в V* (min_df={BEST_MIN_DF} + keep_words): {len(feature_names_kept)}")
print(f"Удалено как редкие: {len(removed_words)}")
print(f"\n[+] Оставшиеся слова сохранены в: {kept_path}")
print(f"[+] Удаленные слова (с частотой) сохранены в: {removed_path}")

print("\n=== ПРОВЕРКА КРИТИЧЕСКИХ СЛОВ В V* ===")
# Проверяем только униграммы, чтобы биграммы не давали ложных срабатываний
critical_words = ['лавина', 'медведь', 'обморожение', 'кровотечение', 'травма', 'помочь', 'спасатель']
kept_unigrams = set([w for w in feature_names_kept if ' ' not in w])

for word in critical_words:
    if word in kept_unigrams:
        print(f"[+] {word} — осталось в словаре.")
    else:
        in_bigrams = any(word in w.split() for w in feature_names_kept if ' ' in w)
        if in_bigrams:
             print(f"[!] {word} — как униграмма потеряно, но встречается в биграммах.")
        else:
             print(f"[-] {word} — ПОТЕРЯНО! (частота: {int(freq_dict.get(word, 0))})")