@echo off
rem Battery Counter - starts the local server and opens the site in the browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment...
  python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
echo.
echo Battery Counter is running at http://127.0.0.1:8300
echo Close this window to stop it.
echo.
start "" http://127.0.0.1:8300
.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8300
