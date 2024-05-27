# Set the project directory to the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

Write-Output "Project directory set to: $scriptDir"

# Ensure the virtual environment activation script exists and activate it
$venvPath = Join-Path -Path $scriptDir -ChildPath "venv"
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    
    # Set the PYTHONPATH to include the src directory
    $env:PYTHONPATH = ".\\projects\\KAN-Stem\\src;.\\projects\\KAN-Stem"
    
    # Start TensorBoard
    Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c", "tensorboard --logdir=checkpoints\runs"
    
    # Keep the script running to allow TensorBoard to continue
    Write-Host "TensorBoard is running. Press Enter to stop..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    
} else {
    Write-Output "Virtual environment activation script not found. Ensure the virtual environment is set up correctly."
}
