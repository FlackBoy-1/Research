import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error, accuracy_score
import os
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# --- 1. ЗАГРУЗКА ДАННЫХ И ПАРАМЕТРОВ ---
corpus_path = os.path.join(DATA_DIR, 'corpus_preprocessed.csv')
if not os.path.exists(corpus_path):
    print("[ОШИБКА] corpus_preprocessed.csv не найден. Запустите этап 2.")
    sys.exit(1)

df = pd.read_csv(corpus_path, sep=';', encoding='utf-8-sig')

def load_int(filename, default=None):
    try:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            return int(f.read().strip())
    except Exception as e:
        if default is not None:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Файл {filename} не найден. Используем дефолт {default}.")
            return default
        print(f"[ОШИБКА] Файл {filename} не найден. Запустите предыдущие этапы.")
        sys.exit(1)

K_STAR = load_int('best_k.txt')
MIN_DF = load_int('best_min_df.txt')

print(f"Параметры: k* = {K_STAR}, min_df* = {MIN_DF}")

# --- 2. ФИЛЬТРАЦИЯ И РАЗДЕЛЕНИЕ ---
if 'is_synthetic' in df.columns:
    df_real = df[df['is_synthetic'] == 0].copy()
else:
    print("[ПРЕДУПРЕЖДЕНИЕ] Колонка is_synthetic не найдена. Синтетика не фильтруется.")
    df_real = df.copy()

counts = df_real['intent'].value_counts()
valid_classes = counts[counts >= 2].index
df_valid = df_real[df_real['intent'].isin(valid_classes)].copy()

case_to_intent = df_valid.groupby('case_id')['intent'].agg(lambda x: x.value_counts().index[0])
unique_cases = case_to_intent.index

try:
    train_cases, test_cases = train_test_split(
        unique_cases, 
        test_size=0.2, 
        random_state=42, 
        stratify=case_to_intent.values
    )
except ValueError as e:
    print(f"[ПРЕДУПРЕЖДЕНИЕ] Стратификация по case_id не удалась ({e}). Используем обычный сплит.")
    train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=42)

train_df = df_valid[df_valid['case_id'].isin(train_cases)].copy()
test_df = df_valid[df_valid['case_id'].isin(test_cases)].copy()

print(f"Train: {len(train_df)} строк (уникальных кейсов: {len(train_cases)})")
print(f"Test:  {len(test_df)} строк (уникальных кейсов: {len(test_cases)})")

# --- 3. ВЕКТОРИЗАЦИЯ (Только на Train!) ---
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

if X_train.shape[1] <= 1:
    print("[ОШИБКА] Словарь пуст или содержит 1 слово. Уменьшите min_df.")
    sys.exit(1)

actual_k = min(K_STAR, X_train.shape[1] - 1)
svd = TruncatedSVD(n_components=actual_k, random_state=42)
X_train_lsa = svd.fit_transform(X_train)
X_test_lsa = svd.transform(X_test)

# --- 4. ОБУЧЕНИЕ КЛАССИФИКАТОРОВ ---
# 4.1 Предсказание интента
clf_intent = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_intent.fit(X_train_lsa, train_df['intent'])
y_pred_intent = clf_intent.predict(X_test_lsa)

# 4.2 Предсказание urgency
clf_urgency = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_urgency.fit(X_train_lsa, train_df['urgency'])
y_pred_urgency = clf_urgency.predict(X_test_lsa)

# 4.3 Единый классификатор для F1_joint (intent + urgency)
train_df['intent_urgency'] = train_df['intent'].astype(str) + "_" + train_df['urgency'].astype(str)
test_df['intent_urgency'] = test_df['intent'].astype(str) + "_" + test_df['urgency'].astype(str)

clf_joint = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_joint.fit(X_train_lsa, train_df['intent_urgency'])
y_pred_joint = clf_joint.predict(X_test_lsa)

# --- 5. РАСЧЕТ МЕТРИК ---
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (РАСШИРЕННЫЙ СПИСОК)")
print("="*50)

# 1. F1_intent
f1_int = f1_score(test_df['intent'], y_pred_intent, average='macro', zero_division=0)
print(f"1. F1_intent (Macro): {f1_int:.3f}")

# 2. F1_urgency + MAE_urgency
f1_urg = f1_score(test_df['urgency'], y_pred_urgency, average='macro', zero_division=0)
mae_urg = mean_absolute_error(test_df['urgency'], y_pred_urgency)
print(f"2. F1_urgency (Macro): {f1_urg:.3f}")
print(f"   MAE_urgency:         {mae_urg:.3f} (средняя ошибка в уровнях)")

# 3. F1_joint (support >= 10) - Строгая
joint_counts = test_df['intent_urgency'].value_counts()
valid_joints_10 = joint_counts[joint_counts >= 10].index
mask_joint_10 = test_df['intent_urgency'].isin(valid_joints_10)

if mask_joint_10.sum() > 0:
    f1_joint_10 = f1_score(
        test_df.loc[mask_joint_10, 'intent_urgency'], 
        y_pred_joint[mask_joint_10], 
        average='macro', zero_division=0
    )
    print(f"3. F1_joint (Macro, support>=10): {f1_joint_10:.3f}")
    print(f"   (учтено пар: {len(valid_joints_10)} из {len(joint_counts)})")
else:
    print("3. F1_joint (support>=10): Недостаточно данных.")
    f1_joint_10 = np.nan

# 4. F1_joint (support >= 3) - Мягкая
valid_joints_3 = joint_counts[joint_counts >= 3].index
mask_joint_3 = test_df['intent_urgency'].isin(valid_joints_3)

if mask_joint_3.sum() > 0:
    f1_joint_3 = f1_score(
        test_df.loc[mask_joint_3, 'intent_urgency'], 
        y_pred_joint[mask_joint_3], 
        average='macro', zero_division=0
    )
    print(f"4. F1_joint (Macro, support>=3):  {f1_joint_3:.3f}")
    print(f"   (учтено пар: {len(valid_joints_3)} из {len(joint_counts)})")
else:
    print("4. F1_joint (support>=3): Недостаточно данных.")
    f1_joint_3 = np.nan

# 5. F1_urgency_on_SOS
sos_mask = test_df['intent'] == 'SOS'
if sos_mask.sum() > 0:
    f1_sos_urg = f1_score(
        test_df.loc[sos_mask, 'urgency'], 
        y_pred_urgency[sos_mask], 
        average='macro', zero_division=0
    )
    print(f"5. F1_urgency_on_SOS (Macro): {f1_sos_urg:.3f}")
else:
    print("5. F1_urgency_on_SOS: В тесте нет SOS-сообщений.")
    f1_sos_urg = np.nan

# 6. Conditional Accuracy: Точность предсказания urgency при условии, что intent предсказан верно
correct_intent_mask = (test_df['intent'] == y_pred_intent)
if correct_intent_mask.sum() > 0:
    cond_acc = accuracy_score(
        test_df.loc[correct_intent_mask, 'urgency'], 
        y_pred_urgency[correct_intent_mask]
    )
    print(f"6. Urgency Acc on Correct Intent: {cond_acc:.3f}")
    print(f"   (посчитано на {correct_intent_mask.sum()} сообщениях с верным интентом)")
else:
    print("6. Urgency Acc on Correct Intent: Нет сообщений с верным интентом.")
    cond_acc = np.nan

# --- 6. СОХРАНЕНИЕ ОТЧЕТОВ ---
val_dir = os.path.join(PLOTS_DIR, 'validation')
os.makedirs(val_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Матрица ошибок для интентов
labels_test = sorted(df_valid['intent'].unique())
cm = confusion_matrix(test_df['intent'], y_pred_intent, labels=labels_test)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_test, yticklabels=labels_test)
plt.title(f'Матрица ошибок (Intent)\nСтрок: {len(df)} | k*={K_STAR} | min_df={MIN_DF}')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_file = os.path.join(val_dir, f'confusion_matrix_intent_{timestamp}.png')
plt.savefig(cm_file, dpi=300)
plt.close()

# Текстовый отчет
report_intent = classification_report(test_df['intent'], y_pred_intent, zero_division=0)
report_urg = classification_report(test_df['urgency'], y_pred_urgency, zero_division=0)
report_joint = classification_report(test_df['intent_urgency'], y_pred_joint, zero_division=0)

report_file = os.path.join(val_dir, f'report_{timestamp}.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f"Параметры: k*={K_STAR}, min_df={MIN_DF}\n")
    f.write(f"Train: {len(train_df)} строк, Test: {len(test_df)} строк\n\n")
    
    f.write("=== СВОДКА МЕТРИК ===\n")
    f.write(f"F1_intent: {f1_int:.3f}\n")
    f.write(f"F1_urgency: {f1_urg:.3f}\n")
    f.write(f"MAE_urgency: {mae_urg:.3f}\n")
    f.write(f"F1_joint (support>=10): {f1_joint_10:.3f}\n")
    f.write(f"F1_joint (support>=3): {f1_joint_3:.3f}\n")
    f.write(f"F1_urgency_on_SOS: {f1_sos_urg:.3f}\n")
    f.write(f"Urgency Acc on Correct Intent: {cond_acc:.3f}\n\n")

    f.write("=== МЕТРИКИ ИНТЕНТОВ ===\n")
    f.write(report_intent + "\n\n")
    
    f.write("=== МЕТРИКИ URGENCY ===\n")
    f.write(report_urg + "\n\n")
    
    f.write("=== МЕТРИКИ JOINT (intent_urgency) ===\n")
    f.write(report_joint)

print(f"\nМатрица ошибок сохранена: {cm_file}")
print(f"Текстовый отчет сохранен: {report_file}")