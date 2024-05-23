# Set source and destination paths
$sourcePath = "K:\KAN-Stem DataSet"
$destinationPath = "K:\KAN-Stem DataSet"

# Function to get a unique file name if there's a conflict
function Get-UniqueFileName {
    param (
        [string]$filePath
    )

    $fileDir = [System.IO.Path]::GetDirectoryName($filePath)
    $fileName = [System.IO.Path]::GetFileNameWithoutExtension($filePath)
    $fileExtension = [System.IO.Path]::GetExtension($filePath)
    $counter = 1

    while (Test-Path $filePath) {
        $filePath = "$fileDir\$fileName($counter)$fileExtension"
        $counter++
    }

    return $filePath
}

# Create the destination folder if it doesn't exist
if (-Not (Test-Path $destinationPath)) {
    New-Item -ItemType Directory -Path $destinationPath
}

# Get all folders in the source directory
$folders = Get-ChildItem -Path $sourcePath -Directory

foreach ($folder in $folders) {
    # Get all files in each subfolder
    $files = Get-ChildItem -Path $folder.FullName -File

    foreach ($file in $files) {
        $destinationFile = Join-Path -Path $destinationPath -ChildPath $file.Name

        if (Test-Path $destinationFile) {
            $destinationFile = Get-UniqueFileName -filePath $destinationFile
        }

        Move-Item -Path $file.FullName -Destination $destinationFile
        Write-Output "Moved: $($file.FullName) to $destinationFile"
    }
}
