# Set the project directory
Set-Location -Path "C:\projects\KAN-Stem"

# Ensure the virtual environment activation script exists and activate it
$venvPath = "C:\projects\KAN-Stem\venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    
    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "C:\projects\KAN-Stem\src;C:\projects\KAN-Stem"
    
    # Run the training script
    python src\train.py
    
    # Launch TensorBoard for performance monitoring
    tensorboard --logdir logs/
} else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
}
