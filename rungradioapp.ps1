param(
    [string]$checkpointDir = "./checkpoints",
    [string]$filePath
)

# Set the project directory to the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

# Ensure the virtual environment activation script exists and activate it
$venvPath = Join-Path -Path $scriptDir -ChildPath "venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript

    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "$scriptDir\src;$scriptDir"

    # Run the gradio_app.py script with arguments
    python .\src\gradio_app.py --checkpoint_dir "$checkpointDir" --file_path "$filePath"
}
else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
}
