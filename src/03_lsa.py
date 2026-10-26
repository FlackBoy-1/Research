import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os
import warnings
import sys
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

print(f"Загружено стоп-слов: {len(stop_words)}")
print(f"Защищено (keep): {len(keep_words)}")
print(f"Итого удаляется: {len(final_stop_words)}")

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

# --- ДИАГНОСТИКА ПУСТЫХ СТРОК ---
# Ищем строки, где lemmas пустое или NaN
empty_mask = df['lemmas'].isna() | (df['lemmas'].astype(str).str.strip() == '')
if empty_mask.any():
    print("\n[КРИТИЧЕСКАЯ ОШИБКА] Найдены строки, которые полностью опустели после предобработки!")
    print("Вероятная причина: текст состоял только из латиницы/цифр, а вы включили USE_LATIN_WORDS = False.")
    print("Пожалуйста, удалите эти строки из data/corpus.csv или исправьте их текст.\n")
    
    bad_rows = df[empty_mask]
    for idx, row in bad_rows.iterrows():
        print(f"  ID: {row.get('id', 'N/A')} | Исходный текст: '{row.get('text', '')}'")
    
    print("\nУдалите эти ID из вашего датасета и перезапустите 02_preprocess.py.")
    sys.exit(1)

# Если пустых строк нет, работаем дальше
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words 
)

X = vectorizer.fit_transform(df['lemmas'])
print(f"\nРазмер исходной матрицы: {X.shape}")

with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)
with open(os.path.join(DATA_DIR, 'matrix_x.pkl'), 'wb') as f:
    pickle.dump(X, f)

svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1), random_state=42)
X_lsa = svd.fit_transform(X)

print(f"\nРазмер сжатой матрицы: {X_lsa.shape}")
print(f"Объясненная дисперсия: {svd.explained_variance_ratio_.sum():.2f}")
print("Матрица и векторизатор сохранены в models/ и data/.")