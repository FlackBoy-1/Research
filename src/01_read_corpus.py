import pandas as pd
import os

# Определяем корневую папку проекта (на уровень выше src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

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