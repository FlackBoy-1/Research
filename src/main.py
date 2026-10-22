import sys
import os
import time
import subprocess

# Включаем UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')

def ask_to_open_plot(plot_subfolder):
    plot_path = os.path.join(BASE_DIR, 'plots', plot_subfolder)
    if not os.path.exists(plot_path):
        return
    ans = input("\n[?] Хотите открыть папку с графиками? (y/n): ").strip().lower()
    if ans == 'y':
        subprocess.Popen(f'explorer "{plot_path}"')

def run_stage(script_name, description, plot_dir=None):
    print(f"\n{'='*60}")
    print(f"ЭТАП: {description}")
    print(f"Скрипт: {script_name}.py")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    script_path = os.path.join(SRC_DIR, f"{script_name}.py")
    
    try:
        # Читаем код скрипта как текст
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        # Создаем словарь с глобальными переменными для скрипта
        script_globals = {
            '__name__': '__main__',
            '__file__': script_path,
            '__builtins__': __builtins__
        }
        
        # Выполняем код
        exec(code, script_globals)
        
    except Exception as e:
        print(f"\n[ОШИБКА] Выполнение {script_name}.py прервано.")
        print(f"Причина: {e}")
        input("Нажмите Enter, чтобы выйти...")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"\n[УСПЕХ] {script_name}.py завершен за {elapsed:.1f} сек.")
    
    if plot_dir:
        ask_to_open_plot(plot_dir)
        
    print("-" * 60)
    input("Нажмите Enter для перехода к следующему этапу...")

if __name__ == "__main__":
    print("ДОБРО ПОЖАЛОВАТЬ В ПАЙПЛАЙН ИССЛЕДОВАНИЯ")
    print("Программа проведет вас по всем этапам обработки данных.\n")
    
    scripts = [
        ('01_read_corpus', '1. Чтение и валидация исходного корпуса', None),
        ('02_preprocess', '2. Очистка текста и лемматизация', None),
        ('03_lsa', '3. Векторизация (TF-IDF) и пробное сжатие (SVD)', None),
        ('04_find_k', '4. Поиск оптимального числа компонент (k*)', 'k_analysis'),
        ('05_select_k', '5. Анализ и выбор k* по правилам', None),
        ('06_find_v', '6. Поиск оптимального размера словаря (V*)', 'v_analysis'),
        ('07_audit_vocab', '7. Аудит словаря (проверка критических слов)', None),
        ('08_validation', '8. Валидация модели на тестовой выборке', 'validation')
    ]
    
    for script_name, desc, plot_dir in scripts:
        run_stage(script_name, desc, plot_dir)
        
    print("\nВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ! Проверьте папку plots.")
    input("Нажмите Enter для выхода...")