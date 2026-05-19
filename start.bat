@echo off
setlocal EnableExtensions

REM Always run from this script's directory (handles spaces safely).
cd /d "%~dp0" || (
	echo Failed to switch to project directory.
	exit /b 1
)

set "PROJECT_DIR=%CD%"
set "ACTIVATE_SCRIPT=%PROJECT_DIR%\.venv\Scripts\activate.bat"

echo Starting Movie Recommender MLOps Pipeline...
echo.

if not exist "%ACTIVATE_SCRIPT%" (
	echo Virtual environment not found at:
	echo   %ACTIVATE_SCRIPT%
	echo Create it first:
	echo   python -m venv .venv
	echo   .venv\Scripts\activate
	echo   pip install -r requirements.txt
	exit /b 1
)

call "%ACTIVATE_SCRIPT%" || (
	echo Failed to activate virtual environment.
	exit /b 1
)

where python >nul 2>&1 || (
	echo Python is not available after venv activation.
	exit /b 1
)

echo Starting MLflow server on port 5000...
start "MLflow Server" cmd /k "cd /d ^"%PROJECT_DIR%^" && call ^"%ACTIVATE_SCRIPT%^" && mlflow server --host 0.0.0.0 --port 5000"

echo Waiting for MLflow to start...
timeout /t 5 /nobreak >nul

echo Training model...
python train.py
if errorlevel 1 (
	echo Training failed. Stopping startup.
	exit /b 1
)

echo Running drift simulation...
python drift_simulation.py
if errorlevel 1 (
	echo Drift simulation failed. Continuing startup.
)

echo Starting FastAPI on port 8000...
start "FastAPI Server" cmd /k "cd /d ^"%PROJECT_DIR%^" && call ^"%ACTIVATE_SCRIPT%^" && uvicorn app.main:app --reload --port 8000"

echo Waiting for FastAPI to start...
timeout /t 5 /nobreak >nul

echo Starting Streamlit on port 8501...
start "Streamlit App" cmd /k "cd /d ^"%PROJECT_DIR%^" && call ^"%ACTIVATE_SCRIPT%^" && streamlit run streamlit_app.py"

echo.
echo Everything is running.
echo.
echo MLflow    : http://localhost:5000
echo FastAPI   : http://localhost:8000
echo Streamlit : http://localhost:8501
echo.
pause