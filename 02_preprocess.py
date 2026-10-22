import pandas as pd
import re
import pymorphy3

# 1. Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()

# 2. Функция для очистки текста
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Функция для лемматизации через pymorphy3
def lemmatize(text):
    words = re.findall(r'[а-яё]+', text)
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)

# 4. Читаем таблицу
df = pd.read_csv('corpus.csv', sep=';', encoding='utf-8-sig')

# 5. Применяем очистку
df['clean_text'] = df['text'].apply(clean_text)

# 6. Применяем лемматизацию
print("Идет лемматизация, подождите...")
df['lemmas'] = df['clean_text'].apply(lemmatize)

# 7. Выводим результат
print("--- Оригинал vs Леммы ---")
for i in range(5):
    print(f"ID {df['id'][i]}:")
    print(f"  До:       {df['text'][i]}")
    print(f"  Очищено:  {df['clean_text'][i]}")
    print(f"  Леммы:    {df['lemmas'][i]}\n")

# 8. Сохраняем результат в новый файл
df.to_csv('corpus_preprocessed.csv', sep=';', index=False, encoding='utf-8-sig')
print("\nРезультат сохранен в файл corpus_preprocessed.csv")