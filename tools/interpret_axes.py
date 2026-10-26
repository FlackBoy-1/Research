import pandas as pd
import pickle
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. Загрузка данных и параметров
print("Загрузка данных...")
df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

try:
    with open(os.path.join(DATA_DIR, 'best_k.txt'), 'r') as f:
        K_STAR = int(f.read().strip())
    print(f"Загружен k* = {K_STAR}")
except:
    K_STAR = 69
    print(f"Файл best_k.txt не найден. Используем k*={K_STAR} по умолчанию.")

try:
    with open(os.path.join(DATA_DIR, 'best_min_df.txt'), 'r') as f:
        BEST_MIN_DF = int(f.read().strip())
    print(f"Загружен min_df = {BEST_MIN_DF}")
except:
    BEST_MIN_DF = 4
    print(f"Файл best_min_df.txt не найден. Используем min_df={BEST_MIN_DF} по умолчанию.")

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# 2. Векторизация с актуальным min_df
print(f"\nПостроение TF-IDF матрицы (min_df={BEST_MIN_DF})...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=BEST_MIN_DF 
)

X = vectorizer.fit_transform(df['lemmas'])
print(f"Размер матрицы: {X.shape} (строк, слов)")

# 3. Обучение SVD
print(f"Обучение SVD (k*={K_STAR})...")
svd = TruncatedSVD(n_components=K_STAR, random_state=42)
svd.fit(X)

# Сохраняем модель SVD с указанием min_df в имени
svd_filename = f'svd_model_min_df_{BEST_MIN_DF}.pkl'
with open(os.path.join(MODELS_DIR, svd_filename), 'wb') as f:
    pickle.dump(svd, f)

feature_names = vectorizer.get_feature_names_out()
components = svd.components_

# 4. Сбор результатов (с указанием min_df в именах файлов)
axes_data = {}
txt_output_path = os.path.join(DATA_DIR, f'axes_top_words_min_df_{BEST_MIN_DF}.txt')
json_output_path = os.path.join(DATA_DIR, f'axes_prompts_min_df_{BEST_MIN_DF}.json')

print("\nИзвлечение топ-слов...")
with open(txt_output_path, 'w', encoding='utf-8') as txt_f:
    txt_f.write(f"ТОП-10 СЛОВ ПО ОСЯМ LSA (всего осей: {K_STAR}, min_df={BEST_MIN_DF})\n")
    txt_f.write("="*60 + "\n\n")
    
    for i in range(K_STAR):
        # Берем индексы 10 наибольших весов по модулю (плюс и минус важны для LSA)
        top_indices = components[i].argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        
        # Получаем объясненную дисперсию для этой оси (в процентах)
        variance_ratio = svd.explained_variance_ratio_[i] * 100
        
        # Запись в TXT
        txt_f.write(f"Ось {i:02d} (дисперсия: {variance_ratio:.4f}%): {', '.join(top_words)}\n")
        
        # Запись в JSON
        words_str = ", ".join(top_words)
        axes_data[f"axis_{i}"] = {
            "variance": variance_ratio,
            "top_words": top_words,
            "prompt": f"Слова: [{words_str}]. Придумай короткое название темы (2-4 слова) строго в контексте ЧС и туризма. Верни только название."
        }

with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(axes_data, f, ensure_ascii=False, indent=4)

print(f"\n[+] Модель SVD сохранена: models/{svd_filename}")
print(f"[+] Топ-слова сохранены: {txt_output_path}")
print(f"[+] JSON с промптами сохранен: {json_output_path}")