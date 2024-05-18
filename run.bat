@echo off
REM Ensure the virtual environment is activated
call venv\Scripts\activate

REM Run the Gradio app
python app.py

REM Deactivate the virtual environment after running the script
deactivate
