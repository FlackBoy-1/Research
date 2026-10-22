import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import os
# Определяем корневую папку проекта (на уровень выше src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

df = pd.read_csv('corpus_preprocessed.csv', sep=';', encoding='utf-8-sig')

def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# Настраиваем векторизатор на V* (min_df=3)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=3  # Наш V*
)

X = vectorizer.fit_transform(df['lemmas'])
feature_names = vectorizer.get_feature_names_out()

print(f"=== СЛОВА, КОТОРЫЕ ОСТАЛИСЬ В V* ({len(feature_names)} шт.) ===")
for i, word in enumerate(feature_names):
    print(f"{i+1}. {word}")

print("\n=== ПРОВЕРКА КРИТИЧЕСКИХ СЛОВ ===")
critical_words = ['лавина', 'медведь', 'обморожение', 'кровотечение', 'травма', 'помочь', 'спасатель']
for word in critical_words:
    if word in feature_names:
        print(f"[+] {word} — осталось в словаре.")
    else:
        print(f"[-] {word} — ПОТЕРЯНО (встречается слишком редко).")