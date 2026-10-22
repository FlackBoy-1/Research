import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Читаем файл из папки data
df = pd.read_csv(
    os.path.join(DATA_DIR, 'corpus.csv'), 
    sep=';', 
    encoding='utf-8-sig'
)

print("--- Первые 5 строк таблицы ---")
print(df.head())

print("\n--- Информация о таблице ---")
print(df.info())

# Проверка наличия case_id
if 'case_id' in df.columns:
    print(f"\n[+] Столбец case_id найден.")
    print(f"Всего строк: {len(df)}")
    print(f"Уникальных сценариев (case_id): {df['case_id'].nunique()}")
else:
    print("\n[!] ВНИМАНИЕ: Столбец case_id не найден! Группировка при валидации работать не будет.")