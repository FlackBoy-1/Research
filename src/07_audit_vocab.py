import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words(os.path.join(DATA_DIR, 'stop_words.txt'))
keep_words = load_words(os.path.join(DATA_DIR, 'keep_words.txt'))
final_stop_words = [word for word in stop_words if word not in keep_words]

# ЧИТАЕМ min_df* ИЗ ФАЙЛА
try:
    with open(os.path.join(DATA_DIR, 'best_min_df.txt'), 'r') as f:
        BEST_MIN_DF = int(f.read().strip())
    print(f"Автоматически загружен min_df* = {BEST_MIN_DF} из файла.")
except:
    BEST_MIN_DF = 3
    print("Файл best_min_df.txt не найден. Используем min_df=3 по умолчанию.")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=BEST_MIN_DF 
)

X = vectorizer.fit_transform(df['lemmas'])
feature_names = vectorizer.get_feature_names_out()

print(f"=== СЛОВА, КОТОРЫЕ ОСТАЛИСЬ В V* ({len(feature_names)} шт.) ===")
for i, word in enumerate(feature_names):
    if i < 100: # Ограничим вывод, чтобы не засорять консоль
        print(f"{i+1}. {word}")
if len(feature_names) > 100:
    print(f"... и еще {len(feature_names) - 100} слов.")

print("\n=== ПРОВЕРКА КРИТИЧЕСКИХ СЛОВ ===")
critical_words = ['лавина', 'медведь', 'обморожение', 'кровотечение', 'травма', 'помочь', 'спасатель']
for word in critical_words:
    if word in feature_names:
        print(f"[+] {word} — осталось в словаре.")
    else:
        print(f"[-] {word} — ПОТЕРЯНО (встречается слишком редко).")