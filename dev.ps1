# Dev shell: put uv on PATH.   Usage:  . .\dev.ps1
$uvDir = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe"
if (Test-Path $uvDir) { $env:Path = "$uvDir;$env:Path" }
if (Test-Path "$HOME\.local\bin") { $env:Path = "$HOME\.local\bin;$env:Path" }
Write-Host "uv ready: $(uv --version)" -ForegroundColor Green
