$ErrorActionPreference = "Continue"

$BoardIp = "192.168.77.2"
$BoardUser = "root"
$RuntimeDir = Join-Path $env:TEMP "edgevision-runtime"
$StatePath = Join-Path $RuntimeDir "launcher-state.json"

function Stop-ProcessTree([int] $ProcessId) {
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        & taskkill.exe /PID $ProcessId /T /F | Out-Null
        Write-Host "Stopped PC process tree $ProcessId"
    }
}

if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    if ($null -ne $state.event_logger_pid) {
        if (Get-Process -Id ([int]$state.event_logger_pid) -ErrorAction SilentlyContinue) {
            Stop-Process -Id ([int]$state.event_logger_pid) -ErrorAction SilentlyContinue
            Write-Host "Stopped event subscriber $($state.event_logger_pid)"
        }
    }
    if ($null -ne $state.bridge_pid) {
        Stop-ProcessTree ([int]$state.bridge_pid)
    }
} else {
    Write-Host "No launcher state file; no tracked PC helpers to stop"
}

$stopCommand = "pkill -TERM -x edge_vision 2>/dev/null || true; sleep 1; if pgrep -x edge_vision >/dev/null; then echo EDGEVISION_STOP_FAILED; exit 1; else echo EDGEVISION_STOPPED; fi"
& ssh.exe -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$BoardUser@$BoardIp" $stopCommand 2>&1 | Out-Host
if ($LASTEXITCODE -eq 0) {
    Write-Host "Board EdgeVision stopped"
} else {
    Write-Warning "Board stop could not be confirmed"
}

if (Test-Path -LiteralPath $StatePath) {
    Remove-Item -LiteralPath $StatePath -Force
}
