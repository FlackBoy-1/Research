import pandas as pd
import pickle
import os
import json
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. Загружаем данные и модели
try:
    with open(os.path.join(DATA_DIR, 'matrix_x.pkl'), 'rb') as f:
        X = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("[ОШИБКА] Не найдены файлы matrix_x.pkl или vectorizer.pkl. Запустите этап 3.")
    exit()

# 2. Динамически читаем актуальное k*
try:
    with open(os.path.join(DATA_DIR, 'best_k.txt'), 'r') as f:
        K_STAR = int(f.read().strip())
    print(f"Автоматически загружен k* = {K_STAR} из файла.")
except:
    K_STAR = 67
    print(f"Файл best_k.txt не найден. Используем k*={K_STAR} по умолчанию.")

# 3. Обучаем SVD (или загружаем, если сохраняли)
print(f"Извлечение топ-слов для {K_STAR} осей...")
svd = TruncatedSVD(n_components=K_STAR, random_state=42)
svd.fit(X)

# Сохраняем SVD на будущее (чтобы проецировать новые тексты)
with open(os.path.join(MODELS_DIR, 'svd_model.pkl'), 'wb') as f:
    pickle.dump(svd, f)

feature_names = vectorizer.get_feature_names_out()
components = svd.components_

# 4. Собираем результаты
axes_data = {}
txt_output_path = os.path.join(DATA_DIR, 'axes_top_words.txt')
json_output_path = os.path.join(DATA_DIR, 'axes_prompts.json')

with open(txt_output_path, 'w', encoding='utf-8') as txt_f:
    txt_f.write(f"ТОП-10 СЛОВ ПО ОСЯМ LSA (всего осей: {K_STAR})\n")
    txt_f.write("="*60 + "\n\n")
    
    for i in range(K_STAR):
        # Берем индексы 10 наибольших весов для текущей оси
        # (для SVD отрицательные веса тоже важны, берем по абсолютной величине)
        top_indices = components[i].argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        
        # Вывод в консоль (кратко)
        print(f"Ось {i:02d}: {', '.join(top_words)}")
        
        # Запись в TXT
        txt_f.write(f"Ось {i:02d}: {', '.join(top_words)}\n")
        
        # Запись в JSON (с готовым промптом для LLM)
        words_str = ", ".join(top_words)
        axes_data[f"axis_{i}"] = {
            "top_words": top_words,
            "prompt": f"Слова: [{words_str}]. Придумай короткое название темы (2-4 слова) строго в контексте ЧС и туризма. Верни только название."
        }

# Сохраняем JSON
with open(json_output_path, 'w', encoding='utf-8') as json_f:
    json.dump(axes_data, json_f, ensure_ascii=False, indent=4)

print(f"\n[+] Модель SVD сохранена: models/svd_model.pkl")
print(f"[+] Топ-слова сохранены: {txt_output_path}")
print(f"[+] JSON с промптами сохранен: {json_output_path}")