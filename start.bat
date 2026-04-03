@echo off
echo Running...
call venv\Scripts\activate
streamlit run app/web/interface.py
pause