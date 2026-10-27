import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================
# НАСТРОЙКИ
# ============================================================
MIN_DF = 4 # Текущий порог
# ============================================================

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

print("Анализ редких слов (кандидатов в keep_words)...")
corpus_path = os.path.join(DATA_DIR, 'corpus_preprocessed.csv')
df = pd.read_csv(corpus_path, sep=';', encoding='utf-8-sig', keep_default_na=False)
df['intent'] = df['intent'].map(MACRO_MAPPING)
df = df.dropna(subset=['intent'])

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

current_keep_words = set(load_words('keep_words.txt'))

# Считаем частоты всех слов
vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=False, norm=None) # Только униграммы для простоты
X = vectorizer.fit_transform(df['lemmas'])
frequencies = X.sum(axis=0).A1
feature_names = vectorizer.get_feature_names_out()

# Создаем словарь: слово -> класс
word_to_class = {}
for i, word in enumerate(feature_names):
    for idx, row in df.iterrows():
        if word in row['lemmas'].split():
            word_to_class[word] = row['intent']
            break

print(f"Всего уникальных слов: {len(feature_names)}")
print(f"Слов с частотой < {MIN_DF} (отбрасываются): {np.sum(frequencies < MIN_DF)}")

print("\n--- ТОП-50 РЕДКИХ СЛОВ, КОТОРЫЕ СТОИТ ДОБАВИТЬ В KEEP_WORDS ---")
print(f"{'Слово':<20} | {'Частота':<8} | {'Класс':<20} | {'Статус'}")
print("-" * 65)

# Собираем кандидатов
candidates = []
for i, word in enumerate(feature_names):
    freq = frequencies[i]
    if freq < MIN_DF:
        # Убираем стоп-слова и мусор
        if len(word) < 3 or word.isdigit():
            continue
            
        cls = word_to_class.get(word, 'unknown')
        candidates.append((word, freq, cls))

# Сортируем по частоте (от большей к меньшей, но меньше 4)
candidates.sort(key=lambda x: x[1], reverse=True)

for word, freq, cls in candidates[:50]:
    status = "[УЖЕ В СПИСКЕ]" if word in current_keep_words else "[НУЖНО ДОБАВИТЬ]"
    print(f"{word:<20} | {int(freq):<8} | {cls:<20} | {status}")

print("\n[ИНСТРУКЦИЯ]:")
print("1. Откройте файл data/keep_words.txt")
print("2. Скопируйте туда слова из списка, у которых статус [НУЖНО ДОБАВИТЬ]")
print("3. Перезапустите src/08_validation.py для проверки метрик.")