@echo off
title NIR LSA Analysis

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ====================================================
echo   STARTING RESEARCH PIPELINE
echo ====================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [WARNING] Virtual environment not found!
    python -m venv venv
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\pip.exe install -r requirements.txt
)

:: Просто запускаем скрипт
venv\Scripts\python.exe src\main.py

echo.
echo Work finished. Press any key...
pause >nul