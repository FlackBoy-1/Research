import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# --- 1. ЗАГРУЗКА ДАННЫХ И ПАРАМЕТРОВ ---
df = pd.read_csv(os.path.join(DATA_DIR, 'corpus_preprocessed.csv'), sep=';', encoding='utf-8-sig')

def load_int(filename, default):
    try:
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            return int(f.read().strip())
    except:
        return default

K_STAR = load_int('best_k.txt', 67)
MIN_DF = load_int('best_min_df.txt', 2)

print(f"Параметры: k* = {K_STAR}, min_df* = {MIN_DF}")

# --- 2. РАЗДЕЛЕНИЕ НА TRAIN/TEST (ГРУППИРОВКА ПО case_id) ---
# Берем только не-синтетику для теста (по правилу 5)
df_real = df[df['is_synthetic'] == 0].copy() if 'is_synthetic' in df.columns else df.copy()

# Убираем классы с 1 примером, чтобы не ломать стратификацию
counts = df_real['intent'].value_counts()
valid_classes = counts[counts >= 2].index
df_valid = df_real[df_real['intent'].isin(valid_classes)].copy()

# Разделяем по case_id, чтобы перефразы не попали в тест
unique_cases = df_valid['case_id'].unique()
train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=42)

train_df = df_valid[df_valid['case_id'].isin(train_cases)]
test_df = df_valid[df_valid['case_id'].isin(test_cases)]

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

actual_k = min(K_STAR, X_train.shape[1] - 1)
svd = TruncatedSVD(n_components=actual_k, random_state=42)
X_train_lsa = svd.fit_transform(X_train)
X_test_lsa = svd.transform(X_test)

# --- 4. ОБУЧЕНИЕ КЛАССИФИКАТОРОВ ---
# 4.1 Предсказание интента
clf_intent = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_intent.fit(X_train_lsa, train_df['intent'])
y_pred_intent = clf_intent.predict(X_test_lsa)

# 4.2 Предсказание urgency (срочности)
clf_urgency = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_urgency.fit(X_train_lsa, train_df['urgency'])
y_pred_urgency = clf_urgency.predict(X_test_lsa)

# --- 5. РАСЧЕТ МЕТРИК ---
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (4 МЕТРИКИ)")
print("="*50)

# 1. F1_intent
f1_int = f1_score(test_df['intent'], y_pred_intent, average='macro', zero_division=0)
print(f"1. F1_intent (Macro): {f1_int:.3f}")

# 2. F1_urgency + MAE_urgency
f1_urg = f1_score(test_df['urgency'], y_pred_urgency, average='macro', zero_division=0)
mae_urg = mean_absolute_error(test_df['urgency'], y_pred_urgency)
print(f"2. F1_urgency (Macro): {f1_urg:.3f}")
print(f"   MAE_urgency:         {mae_urg:.3f} (средняя ошибка в уровнях)")

# 3. F1_joint (по парам intent_urgency, min_support=10)
test_df['intent_urgency'] = test_df['intent'].astype(str) + "_" + test_df['urgency'].astype(str)
y_pred_joint = [f"{i}_{u}" for i, u in zip(y_pred_intent, y_pred_urgency)]

joint_counts = test_df['intent_urgency'].value_counts()
valid_joints = joint_counts[joint_counts >= 10].index
mask_joint = test_df['intent_urgency'].isin(valid_joints)

if mask_joint.sum() > 0:
    f1_joint = f1_score(
        test_df.loc[mask_joint, 'intent_urgency'], 
        np.array(y_pred_joint)[mask_joint], 
        average='macro', zero_division=0
    )
    print(f"3. F1_joint (Macro, support>=10): {f1_joint:.3f}")
    print(f"   (учтено пар: {len(valid_joints)} из {len(joint_counts)})")
else:
    print("3. F1_joint: Недостаточно данных для пар с support>=10.")

# 4. F1_urgency_on_SOS
sos_mask = test_df['intent'] == 'SOS'
if sos_mask.sum() > 0:
    f1_sos_urg = f1_score(
        test_df.loc[sos_mask, 'urgency'], 
        np.array(y_pred_urgency)[sos_mask], 
        average='macro', zero_division=0
    )
    print(f"4. F1_urgency_on_SOS (Macro): {f1_sos_urg:.3f}")
else:
    print("4. F1_urgency_on_SOS: В тесте нет SOS-сообщений.")

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

# Текстовый отчет (Classification Report) для интентов
report_intent = classification_report(test_df['intent'], y_pred_intent, zero_division=0)
report_urg = classification_report(test_df['urgency'], y_pred_urgency, zero_division=0)

report_file = os.path.join(val_dir, f'report_{timestamp}.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f"Параметры: k*={K_STAR}, min_df={MIN_DF}\n")
    f.write(f"Train: {len(train_df)} строк, Test: {len(test_df)} строк\n\n")
    f.write("=== МЕТРИКИ ИНТЕНТОВ ===\n")
    f.write(report_intent + "\n\n")
    f.write("=== МЕТРИКИ URGENCY ===\n")
    f.write(report_urg)

print(f"\nМатрица ошибок сохранена: {cm_file}")
print(f"Текстовый отчет сохранен: {report_file}")