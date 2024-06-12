@echo off
echo Checking for existing virtual environment...

REM Check if the venv directory exists
if exist venv (
    echo Existing virtual environment found.

    REM Check the Python version in the virtual environment
    call venv\Scripts\python -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Existing virtual environment is not using Python 3.12. Deleting it...
        rmdir /s /q venv
        if %ERRORLEVEL% NEQ 0 (
            echo Error: Failed to delete existing virtual environment.
            pause
            exit /b 1
        )
        goto createVenv
    ) else (
        echo Existing virtual environment is using Python 3.12. Activating it...
        call venv\Scripts\activate
        if %ERRORLEVEL% NEQ 0 (
            echo Error: Failed to activate virtual environment.
            pause
            exit /b 1
        )
        goto installDeps
    )
) else (
    goto createVenv
)

:createVenv
echo Setting up new virtual environment with Python 3.12...

REM Check if Python 3.12 is installed and get the path
for /f "tokens=*" %%p in ('where python') do (
    set PYTHON_PATH=%%p
    goto :foundPython
)

echo Python not found in PATH. Please install Python or add it to your PATH.
pause
exit /b 1

:foundPython
echo Python found at %PYTHON_PATH%

echo Creating new virtual environment with Python 3.12...
%PYTHON_PATH% -m venv venv

IF %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

IF %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

:installDeps
echo Installing dependencies...
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo Setup complete. To activate the virtual environment, run:
echo call venv\Scripts\activate
pause
