import pandas as pd
import re
import pymorphy3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================
# НАСТРОЙКИ ОБРАБОТКИ
# ============================================================
USE_LATIN_WORDS = False  # True = сохраняем латинские слова, False = удаляем
# ============================================================

morph = pymorphy3.MorphAnalyzer()

def clean_text(text):
    text = str(text).lower()
    
    # 1. Поиск координат (до удаления цифр)
    text = re.sub(r'\b\d{1,3}[.,]\d{3,}\b', ' [coord] ', text)
    text = re.sub(r'\d{1,3}\s*°\s*\d{1,2}', ' [coord] ', text)
    
    # 2. Токен SOS заменяется всегда (безусловно)
    text = re.sub(r'\bsos\b', ' [sos] ', text)
    
    # 3. Обработка латиницы (кроме токенов в скобках)
    if USE_LATIN_WORDS:
        text = re.sub(r'[^а-яёa-z\s.,\[\]]', '', text)
    else:
        text = re.sub(r'[^а-яё\s.,\[\]]', '', text)
    
    # 4. Очистка от лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text):
    # Ищем кириллицу, латиницу и токены в скобках
    words = re.findall(r'\[[a-z]+\]|[а-яё]+|[a-z]+', text)
    lemmas = []
    for word in words:
        if word.startswith('['):
            lemmas.append(word)
        elif re.match(r'[а-яё]+', word):
            lemmas.append(morph.parse(word)[0].normal_form)
        else:
            # Латинские слова оставляем как есть
            lemmas.append(word)
    return ' '.join(lemmas)

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus.csv'), sep=';', encoding='utf-8-sig')

print("Идет очистка текста...")
df['clean_text'] = df['text'].apply(clean_text)

print("Идет лемматизация, подождите...")
df['lemmas'] = df['clean_text'].apply(lemmatize)

print("--- Оригинал vs Леммы ---")
for i in range(5):
    print(f"ID {df['id'][i]}:")
    print(f"  До:       {df['text'][i]}")
    print(f"  Очищено:  {df['clean_text'][i]}")
    print(f"  Леммы:    {df['lemmas'][i]}\n")

if 'case_id' in df.columns:
    print(f"--- Статистика корпуса ---")
    print(f"Всего сообщений: {len(df)}")
    print(f"Уникальных сценариев (case_id): {df['case_id'].nunique()}")

df.to_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', index=False, encoding='utf-8-sig')
print("\nРезультат сохранен в data/corpus_preprocessed.csv")