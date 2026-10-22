import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

import os
# Определяем корневую папку проекта (на уровень выше src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# 1. Загружаем стоп-слова из файлов
def load_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        # Читаем строки, убираем пробелы по краям и пустые строки
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')

# 2. Убираем из стоп-слов те, что в keep_words (защищаем важные слова)
final_stop_words = [word for word in stop_words if word not in keep_words]

print(f"Загружено стоп-слов: {len(stop_words)}")
print(f"Защищено (keep): {len(keep_words)}")
print(f"Итого удаляется: {len(final_stop_words)}")

# 3. Читаем предобработанный файл
df = pd.read_csv('corpus_preprocessed.csv', sep=';', encoding='utf-8-sig')

# 4. Настраиваем векторизатор
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words  # Передаем список стоп-слов
)

# 5. Превращаем текст в матрицу
X = vectorizer.fit_transform(df['lemmas'])
print(f"\nРазмер исходной матрицы: {X.shape}")

# 6. Сохраняем векторизатор и матрицу
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('matrix_x.pkl', 'wb') as f:
    pickle.dump(X, f)

# 7. Обучаем TruncatedSVD (пробный запуск)
svd = TruncatedSVD(n_components=5, random_state=42)
X_lsa = svd.fit_transform(X)

print(f"\nРазмер сжатой матрицы: {X_lsa.shape}")
print(f"Сингулярные значения (первые 5): {svd.singular_values_}")
print(f"Объясненная дисперсия: {svd.explained_variance_ratio_.sum():.2f}")