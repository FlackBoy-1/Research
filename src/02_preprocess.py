import pandas as pd
import re
import pymorphy3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

morph = pymorphy3.MorphAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text):
    words = re.findall(r'[а-яё]+', text)
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus.csv'), sep=';', encoding='utf-8-sig')

df['clean_text'] = df['text'].apply(clean_text)

print("Идет лемматизация, подождите...")
df['lemmas'] = df['clean_text'].apply(lemmatize)

print("--- Оригинал vs Леммы ---")
for i in range(5):
    print(f"ID {df['id'][i]}:")
    print(f"  До:       {df['text'][i]}")
    print(f"  Очищено:  {df['clean_text'][i]}")
    print(f"  Леммы:    {df['lemmas'][i]}\n")

# Статистика по сценариям для журнала
if 'case_id' in df.columns:
    print(f"--- Статистика корпуса ---")
    print(f"Всего сообщений: {len(df)}")
    print(f"Уникальных сценариев (case_id): {df['case_id'].nunique()}")
    print(f"(Помни: число сообщений не равно числу независимых сценариев)")

df.to_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', index=False, encoding='utf-8-sig')
print("\nРезультат сохранен в data/corpus_preprocessed.csv")