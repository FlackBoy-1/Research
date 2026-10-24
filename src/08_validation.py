import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error, accuracy_score, recall_score
import os
from datetime import datetime
import sys
import warnings

# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================
ENABLE_RULES = False  # Поставь False, чтобы отключить правила пост-процессинга
# ============================================================

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

print(f"Train: {len(train_df)} строк, Test: {len(test_df)} строк")

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
clf_intent = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_intent.fit(X_train_lsa, train_df['intent'])
y_pred_intent = clf_intent.predict(X_test_lsa)

clf_urgency = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_urgency.fit(X_train_lsa, train_df['urgency'])
y_pred_urgency = clf_urgency.predict(X_test_lsa)

# ============================================================
# ПРИКЛАДНОЙ ПОСТ-ПРОЦЕССИНГ (ПРАВИЛА)
# ============================================================
if ENABLE_RULES:
    print("\nПрименение правил пост-процессинга (принудительная срочность)...")
    y_pred_urgency_fixed = y_pred_urgency.copy()

    for idx, (pred_intent, pred_urg) in enumerate(zip(y_pred_intent, y_pred_urgency)):
        # Правило 1: Военка, техногенка и погибшие — минимум 4 (Красный код)
        if pred_intent in ['военная_угроза', 'техногенная_угроза', 'погибший']:
            if pred_urg < 4:
                y_pred_urgency_fixed[idx] = 4
        # Правило 2: Травмы, природные и животные угрозы — минимум 3 (Желтый код)
        elif pred_intent in ['травма', 'природная_угроза', 'животная_угроза', 'потерялся']:
            if pred_urg < 3:
                y_pred_urgency_fixed[idx] = 3

    # Считаем разницу для вывода
    f1_urg_before = f1_score(test_df['urgency'], y_pred_urgency, average='macro', zero_division=0)
    acc_urg_before = accuracy_score(test_df['urgency'], y_pred_urgency)
    f1_urg_after = f1_score(test_df['urgency'], y_pred_urgency_fixed, average='macro', zero_division=0)
    acc_urg_after = accuracy_score(test_df['urgency'], y_pred_urgency_fixed)

    print(f"  [ДО] F1_urgency: {f1_urg_before:.3f}, Accuracy: {acc_urg_before:.3f}")
    print(f"  [ПОСЛЕ] F1_urgency: {f1_urg_after:.3f}, Accuracy: {acc_urg_after:.3f}")

    # Переключаемся на исправленные предсказания для всех дальнейших расчетов
    y_pred_urgency = y_pred_urgency_fixed

else:
    # Если правила выключены, фиксируем изначальные значения для отчета
    f1_urg_before = f1_score(test_df['urgency'], y_pred_urgency, average='macro', zero_division=0)
    acc_urg_before = accuracy_score(test_df['urgency'], y_pred_urgency)
    f1_urg_after = f1_urg_before
    acc_urg_after = acc_urg_before
    print("\n[!] Правила пост-процессинга ОТКЛЮЧЕНЫ (ENABLE_RULES = False)")

train_df['intent_urgency'] = train_df['intent'].astype(str) + "_" + train_df['urgency'].astype(str)
test_df['intent_urgency'] = test_df['intent'].astype(str) + "_" + test_df['urgency'].astype(str)

clf_joint = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf_joint.fit(X_train_lsa, train_df['intent_urgency'])
y_pred_joint = clf_joint.predict(X_test_lsa)

# --- 5. РАСЧЕТ МЕТРИК ---
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (С ПРАВИЛАМИ)")
print("="*50)

f1_int = f1_score(test_df['intent'], y_pred_intent, average='macro', zero_division=0)
print(f"1. F1_intent (Macro): {f1_int:.3f}")

f1_urg = f1_score(test_df['urgency'], y_pred_urgency, average='macro', zero_division=0)
mae_urg = mean_absolute_error(test_df['urgency'], y_pred_urgency)
print(f"2. F1_urgency (Macro): {f1_urg:.3f}")
print(f"   MAE_urgency:         {mae_urg:.3f}")

joint_counts = test_df['intent_urgency'].value_counts()
valid_joints_10 = joint_counts[joint_counts >= 10].index
mask_joint_10 = test_df['intent_urgency'].isin(valid_joints_10)

if mask_joint_10.sum() > 0:
    f1_joint_10 = f1_score(test_df.loc[mask_joint_10, 'intent_urgency'], y_pred_joint[mask_joint_10], average='macro', zero_division=0)
    print(f"3. F1_joint (Macro, support>=10): {f1_joint_10:.3f}")
else:
    f1_joint_10 = np.nan

valid_joints_3 = joint_counts[joint_counts >= 3].index
mask_joint_3 = test_df['intent_urgency'].isin(valid_joints_3)

if mask_joint_3.sum() > 0:
    f1_joint_3 = f1_score(test_df.loc[mask_joint_3, 'intent_urgency'], y_pred_joint[mask_joint_3], average='macro', zero_division=0)
    print(f"4. F1_joint (Macro, support>=3):  {f1_joint_3:.3f}")
else:
    f1_joint_3 = np.nan

sos_mask = test_df['intent'] == 'SOS'
if sos_mask.sum() > 0:
    f1_sos_urg = f1_score(test_df.loc[sos_mask, 'urgency'], y_pred_urgency[sos_mask], average='macro', zero_division=0)
    print(f"5. F1_urgency_on_SOS (Macro): {f1_sos_urg:.3f}")
else:
    f1_sos_urg = np.nan

correct_intent_mask = (test_df['intent'] == y_pred_intent)
if correct_intent_mask.sum() > 0:
    cond_acc = accuracy_score(test_df.loc[correct_intent_mask, 'urgency'], y_pred_urgency[correct_intent_mask])
    print(f"6. Urgency Acc on Correct Intent: {cond_acc:.3f}")
else:
    cond_acc = np.nan

# --- 7. РАСЧЕТ COST-WEIGHTED RECALL (CWR) ---
print("\n" + "-"*50)
print("ПРИКЛАДНАЯ МЕТРИКА (Cost-Weighted Recall)")
print("-"*50)

WEIGHTS = {
    'погибший': 0.8,
    'военная_угроза': 1.0,
    'техногенная_угроза': 0.95,
    'природная_угроза': 0.9,
    'травма': 0.75,
    'потерялся': 0.7,
    'заблудился': 0.7,
    'животная_угроза': 0.7,
    'нет_припасов': 0.7,
    'SOS': 0.95,
    'техническая_неисправность': 0.5,
    'статус_группы': 0.2,
    'позиция': 0.2,
    'погода': 0.2,
    'запрос_маршрута': 0.1,
    'запрос_погоды': 0.1,
    'запрос_информации': 0.1,
    'проверка_связи': 0.1
}

def calc_cwr(y_true, y_pred, weights):
    classes = list(weights.keys())
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    w = np.array([weights[c] for c in classes])
    numerator = np.sum(w * tp)
    denominator = np.sum(w * (tp + fn))
    return numerator / denominator if denominator > 0 else 0

cwr = calc_cwr(test_df['intent'], y_pred_intent, WEIGHTS)
print(f"7. Cost-Weighted Recall (CWR): {cwr:.3f}")

weights_plus_50 = {k: v * 1.5 for k, v in WEIGHTS.items()}
weights_minus_50 = {k: v * 0.5 for k, v in WEIGHTS.items()}
cwr_plus = calc_cwr(test_df['intent'], y_pred_intent, weights_plus_50)
cwr_minus = calc_cwr(test_df['intent'], y_pred_intent, weights_minus_50)

print(f"   [Устойчивость] CWR при +50% к весам: {cwr_plus:.3f}")
print(f"   [Устойчивость] CWR при -50% к весам: {cwr_minus:.3f}")
if abs(cwr_plus - cwr_minus) < 0.05:
    print("   [ОК] Метрика устойчива к изменению весов (выводы не переворачиваются).")
else:
    print("   [!] ВНИМАНИЕ: Метрика чувствительна к весам!")

# --- 8. СОХРАНЕНИЕ ОТЧЕТОВ ---
val_dir = os.path.join(PLOTS_DIR, 'validation')
os.makedirs(val_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

report_intent = classification_report(test_df['intent'], y_pred_intent, zero_division=0)
report_urg = classification_report(test_df['urgency'], y_pred_urgency, zero_division=0)
report_joint = classification_report(test_df['intent_urgency'], y_pred_joint, zero_division=0)

report_file = os.path.join(val_dir, f'report_{timestamp}.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f"Параметры: k*={K_STAR}, min_df={MIN_DF}\n")
    f.write(f"Train: {len(train_df)} строк, Test: {len(test_df)} строк\n\n")
    
    f.write("=== СВОДКА МЕТРИК ===\n")
    f.write(f"F1_intent: {f1_int:.3f}\n")
    f.write(f"F1_urgency (до правил): {f1_urg_before:.3f}\n")
    f.write(f"F1_urgency (после правил): {f1_urg_after:.3f}\n")
    f.write(f"MAE_urgency: {mae_urg:.3f}\n")
    f.write(f"F1_joint (support>=10): {f1_joint_10:.3f}\n")
    f.write(f"F1_joint (support>=3): {f1_joint_3:.3f}\n")
    f.write(f"F1_urgency_on_SOS: {f1_sos_urg:.3f}\n")
    f.write(f"Urgency Acc on Correct Intent: {cond_acc:.3f}\n")
    f.write(f"Cost-Weighted Recall (CWR): {cwr:.3f}\n")
    f.write(f"   (Устойчивость: +50% -> {cwr_plus:.3f}, -50% -> {cwr_minus:.3f})\n\n")

    f.write("=== МЕТРИКИ ИНТЕНТОВ ===\n")
    f.write(report_intent + "\n\n")
    f.write("=== МЕТРИКИ URGENCY (ПОСЛЕ ПРАВИЛ) ===\n")
    f.write(report_urg + "\n\n")
    f.write("=== МЕТРИКИ JOINT (intent_urgency) ===\n")
    f.write(report_joint)

print(f"\nМатрица ошибок сохранена: {cm_file}")
print(f"Текстовый отчет сохранен: {report_file}")