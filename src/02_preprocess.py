import pandas as pd
import re
import pymorphy3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================
# НАСТРОЙКИ ОБРАБОТКИ
# ============================================================
USE_LATIN = True  # Поставь False, если латиница начнет мешать
# ============================================================

morph = pymorphy3.MorphAnalyzer()

def clean_text(text):
    text = str(text).lower()
    
    # 1. Поиск координат (до удаления цифр)
    # Ищем числа с точками/запятыми (например 45.123, 38.456) или градусы/минуты
    text = re.sub(r'\b\d{1,3}[.,]\d{3,}\b', ' [coord] ', text)
    text = re.sub(r'\d{1,3}\s*°\s*\d{1,2}', ' [coord] ', text)
    
    # 2. Обработка латиницы (SOS, GPS, LBS и т.д.)
    if USE_LATIN:
        # Заменяем аббревиатуру SOS на токен [sos]
        text = re.sub(r'\bsos\b', ' [sos] ', text)
        # Оставляем латиницу, но удаляем остальную пунктуацию
        text = re.sub(r'[^а-яёa-z\s.,\[\]]', '', text)
    else:
        # Удаляем всю латиницу, кроме токенов в скобках
        text = re.sub(r'[^а-яё\s.,\[\]]', '', text)
    
    # 3. Очистка от лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text):
    # Ищем только кириллицу и токены в квадратных скобках
    # (латиница не лемматизируется pymorphy3, она остается как есть)
    words = re.findall(r'\[[a-z]+\]|[а-яё]+', text)
    lemmas = []
    for word in words:
        if word.startswith('['):
            # Токены [coord], [sos] оставляем без изменений
            lemmas.append(word)
        else:
            # Обычные русские слова лемматизируем
            lemmas.append(morph.parse(word)[0].normal_form)
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

# Статистика по сценариям для журнала
if 'case_id' in df.columns:
    print(f"--- Статистика корпуса ---")
    print(f"Всего сообщений: {len(df)}")
    print(f"Уникальных сценариев (case_id): {df['case_id'].nunique()}")
    print(f"(Помни: число сообщений не равно числу независимых сценариев)")

df.to_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', index=False, encoding='utf-8-sig')
print("\nРезультат сохранен в data/corpus_preprocessed.csv")