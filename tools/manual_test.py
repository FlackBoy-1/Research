import pandas as pd
import numpy as np
import re
import os
import sys
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# НАСТРОЙКИ (должны совпадать с 08_validation.py)
# ============================================================
MIN_DF = 4
WEIGHT_BREAK_1 = 0.6
WEIGHT_BREAK_2 = 1.0
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

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

WEIGHTS = {
    'угроза_военная': 0.8,
    'угроза_техногенная': 0.8,
    'угроза_общая': 1.0,
    'не_угроза': 0.2
}

# --- ФУНКЦИИ ПРЕДОБРАБОТКИ (из 02_preprocess.py) ---
morph = pymorphy3.MorphAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\b\d{1,3}[.,]\d{3,}\b', ' [coord] ', text)
    text = re.sub(r'\d{1,3}\s*°\s*\d{1,2}', ' [coord] ', text)
    text = re.sub(r'\bsos\b', ' [sos] ', text)
    text = re.sub(r'[^а-яё\s.,\[\]]', '', text) # USE_LATIN_WORDS = False
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text):
    words = re.findall(r'\[[a-z]+\]|[а-яё]+', text)
    lemmas = []
    for word in words:
        if word.startswith('['):
            lemmas.append(word)
        else:
            lemmas.append(morph.parse(word)[0].normal_form)
    return ' '.join(lemmas)

# --- ЗАГРУЗКА И ОБУЧЕНИЕ ---
print("Загрузка данных и обучение модели (займет пару секунд)...")
corpus_path = os.path.join(DATA_DIR, 'corpus_preprocessed.csv')
if not os.path.exists(corpus_path):
    print("[ОШИБКА] corpus_preprocessed.csv не найден.")
    sys.exit(1)

df = pd.read_csv(corpus_path, sep=';', encoding='utf-8-sig', keep_default_na=False)
df['intent'] = df['intent'].map(MACRO_MAPPING)
df = df.dropna(subset=['intent'])

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

# Векторизация с защитой keep_words
count_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, use_idf=False, norm=None)
count_matrix = count_vec.fit_transform(df['lemmas'])
frequencies = count_matrix.sum(axis=0).A1
feature_names = count_vec.get_feature_names_out()
valid_features = [w for w, f in zip(feature_names, frequencies) if f >= MIN_DF or w in keep_words]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=final_stop_words, vocabulary=valid_features)
X = vectorizer.fit_transform(df['lemmas'])

# Расчет весов
intent_counts = df['intent'].value_counts()
max_count = intent_counts.max()
cw_intent = {}
for cls, count in intent_counts.items():
    raw_weight = np.sqrt(max_count / count) * WEIGHTS.get(cls, 0.1)
    if raw_weight < WEIGHT_BREAK_1:
        final_weight = WEIGHT_BREAK_1 + (raw_weight / WEIGHT_BREAK_1) * (WEIGHT_BREAK_2 - WEIGHT_BREAK_1)
    elif raw_weight < WEIGHT_BREAK_2:
        final_weight = WEIGHT_BREAK_2 + ((raw_weight - WEIGHT_BREAK_1) / (WEIGHT_BREAK_2 - WEIGHT_BREAK_1)) * 0.2
    else:
        final_weight = raw_weight
    cw_intent[cls] = final_weight

# Обучение
clf_intent = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw_intent)
clf_intent.fit(X, df['intent'])

clf_urgency = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_urgency.fit(X, df['urgency'])

print("Модель готова! Введите 'q' для выхода.\n")

# --- ИНТЕРАКТИВНЫЙ ЦИКЛ ---
while True:
    print("-" * 60)
    user_input = input("Введите текст сообщения (или 'q'): ")
    if user_input.strip().lower() in ['q', 'й']:
        break
    if not user_input.strip():
        continue
    
    # Предобработка
    cleaned = clean_text(user_input)
    lemmas = lemmatize(cleaned)
    
    # Векторизация и предсказание
    X_test = vectorizer.transform([lemmas])
    pred_intent = clf_intent.predict(X_test)[0]
    probs_intent = clf_intent.predict_proba(X_test)[0]
    pred_urgency = clf_urgency.predict(X_test)[0]
    probs_urgency = clf_urgency.predict_proba(X_test)[0]
    
    # Вывод
    print("\n--- РЕЗУЛЬТАТ АНАЛИЗА ---")
    print(f"[*] Исходный текст: {user_input}")
    print(f"[*] Очищенный:      {cleaned}")
    print(f"[*] Леммы:          {lemmas}")
    print(f"\n[!] ИНТЕНТ:    {pred_intent.upper()}")
    
    # Сортируем вероятности по убыванию
    sorted_idx = np.argsort(probs_intent)[::-1]
    for i in sorted_idx:
        cls_name = clf_intent.classes_[i]
        prob = probs_intent[i]
        if prob > 0.01: # Показываем только значимые
            print(f"    - {cls_name}: {prob:.3f}")
            
    print(f"\n[!] СРОЧНОСТЬ: {pred_urgency}")
    sorted_idx_urg = np.argsort(probs_urgency)[::-1]
    for i in sorted_idx_urg:
        cls_name = clf_urgency.classes_[i]
        prob = probs_urgency[i]
        if prob > 0.01:
            print(f"    - Уровень {cls_name}: {prob:.3f}")