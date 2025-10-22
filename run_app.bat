@echo off
echo Launching UTC Factbook State Map...
echo.

REM Create a virtual environment if it doesn’t exist
if not exist venv (
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install dependencies (quietly)
pip install -r requirements.txt >nul

REM Run the Streamlit app
streamlit run app.py

REM Keep the window open after closing
pause
