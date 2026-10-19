# Installs the Sentinel as a background task that starts when you sign in to Windows.
#
# *** NOT TESTED ON WINDOWS. *** It was written against the documented PowerShell cmdlets while the rest of the Sentinel was tested on Linux.
# Read it before running it, run it from a normal (non-administrator) PowerShell in the repository folder, and tell me what breaks.
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install_sentinel.ps1
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install_sentinel.ps1 -Uninstall
#
# It runs the Sentinel as YOU (an ordinary account, not Administrator), so the Sentinel can see only what you can see and has no rights to change
# the system. The Sentinel's memory lives in %LOCALAPPDATA%\Sentinel. Uninstalling removes the scheduled task and leaves the data folder for you to delete.
param(
  [string]$Python = "python",
  [string]$DataDir = "$env:LOCALAPPDATA\Sentinel",
  [int]$Port = 8787,
  [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$Task = "Sentinel"
if ($Uninstall) {
  Unregister-ScheduledTask -TaskName $Task -Confirm:$false
  Write-Host "Task removed. Your data is still in $DataDir; delete it yourself if you want."
  exit 0
}
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$Venv = "$DataDir\venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) { & $Python -m venv $Venv }
& "$Venv\Scripts\python.exe" -m pip install --quiet -r "$Repo\deploy\requirements-sentinel.txt"
$Home_ = "$DataDir\home"
& "$Venv\Scripts\python.exe" -m sentinel init --home $Home_ --port $Port
$Exe = "$Venv\Scripts\pythonw.exe"                       # pythonw: no console window
$Action = New-ScheduledTaskAction -Execute $Exe -Argument "-m sentinel run --home `"$Home_`"" -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $Task -Action $Action -Trigger $Trigger -Settings $Settings -Description "Read-only, alert-only host watcher" -RunLevel Limited | Out-Null
Start-ScheduledTask -TaskName $Task
Write-Host "Sentinel installed and started. Dashboard on this PC: http://127.0.0.1:$Port/"
Write-Host "Test it with the fire drill:  $Venv\Scripts\python.exe $Repo\deploy\sentinel_drill.py --dir `"$env:USERPROFILE\Downloads`""
Write-Host "Kill switch: create an empty file named PAUSE in $Home_ (delete it to resume)."
