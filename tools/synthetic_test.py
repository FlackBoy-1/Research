import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_int(filename, default):
    try:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            return int(f.read().strip())
    except Exception:
        return default

K_STAR = load_int('best_k.txt', 51)
MIN_DF = load_int('best_min_df.txt', 6)

df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

# Берем абсолютно все данные (и реальные, и синтетические)
# Не фильтруем по is_synthetic на этапе сплита
df_valid = df.copy()

# Делаем разделение по case_id, как в основном пайплайне, чтобы не было перефразов
unique_cases = df_valid['case_id'].unique()
train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=42)

train_df = df_valid[df_valid['case_id'].isin(train_cases)].copy()
test_df = df_valid[df_valid['case_id'].isin(test_cases)].copy()

print(f"Train (все данные): {len(train_df)} строк (уникальных кейсов: {len(train_cases)})")
print(f"Test (все данные):  {len(test_df)} строк (уникальных кейсов: {len(test_cases)})")

# Смотрим, сколько синтетики попало в тест
if 'is_synthetic' in test_df.columns:
    synth_in_test = test_df['is_synthetic'].sum()
    print(f"[!] Синтетических сообщений в тесте: {synth_in_test} ({synth_in_test/len(test_df)*100:.1f}%)")

def load_words(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words = load_words('stop_words.txt')
keep_words = load_words('keep_words.txt')
final_stop_words = [word for word in stop_words if word not in keep_words]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    token_pattern=r'(?u)\b\w+\b',
    stop_words=final_stop_words,
    min_df=MIN_DF
)

X_train = vectorizer.fit_transform(train_df['lemmas'])
X_test = vectorizer.transform(test_df['lemmas'])

actual_k = min(K_STAR, X_train.shape[1] - 1)
svd = TruncatedSVD(n_components=actual_k, random_state=42)
X_train_lsa = svd.fit_transform(X_train)
X_test_lsa = svd.transform(X_test)

clf_intent = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_intent.fit(X_train_lsa, train_df['intent'])
y_pred_intent = clf_intent.predict(X_test_lsa)

f1 = f1_score(test_df['intent'], y_pred_intent, average='macro', zero_division=0)
print("\n" + "="*50)
print("РЕЗУЛЬТАТ НА СМЕШАННОМ ТЕСТЕ (С СИНТЕТИКОЙ)")
print("="*50)
print(f"F1_intent (Macro): {f1:.3f}")

print("\nДетальный отчет:")
print(classification_report(test_df['intent'], y_pred_intent, zero_division=0))