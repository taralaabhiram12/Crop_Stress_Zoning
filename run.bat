@echo off
cd /d "%~dp0"
if not exist venv (python -m venv venv & venv\Scripts\python -m pip install -r requirements.txt)
if "%AGRONOMIST_INVITE_CODE%"=="" set AGRONOMIST_INVITE_CODE=AGRO-DEMO
venv\Scripts\python -m streamlit run app.py
pause
