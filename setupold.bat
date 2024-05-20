@echo off
echo Setting up virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install numpy librosa tensorflow

echo Setup complete. To activate the virtual environment, run:
echo call venv\Scripts\activate
pause
