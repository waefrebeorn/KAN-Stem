# Set the project directory
Set-Location -Path "C:\\projects\\KAN-Stem"

# Ensure the virtual environment activation script exists and activate it
$venvPath = "C:\\projects\\KAN-Stem\\venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    
    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = "C:\\projects\\KAN-Stem\\src;C:\\projects\\KAN-Stem"
    
    # Start TensorBoard
    Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c", "tensorboard --logdir=logs"
    
    # Keep the script running to allow TensorBoard to continue
    Write-Host "TensorBoard is running. Press Enter to stop..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    
    # Deactivate the virtual environment
    & "C:\\projects\\KAN-Stem\\venv\\Scripts\\deactivate.ps1"
} else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
}
