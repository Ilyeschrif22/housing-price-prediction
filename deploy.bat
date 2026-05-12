@echo off
REM Quick local run script (Windows)
REM Usage: deploy.bat

echo.
echo Real Estate Price Prediction API - Local Setup
echo ============================================================
echo.

python --version
echo.

echo Installing dependencies...
pip install -r requirements.txt
echo.

echo Training the model...
python train_model.py
echo.

echo Starting local server (Ctrl+C to stop)...
echo Test at: http://localhost:5000/health
echo.

python app.py
pause

