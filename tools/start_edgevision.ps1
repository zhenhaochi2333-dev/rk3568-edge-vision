$ErrorActionPreference = "Stop"

# Formal Windows one-command launcher. It orchestrates the already-validated
# board executable, C++ PC bridge, and C++ event subscriber.
$BoardIp = "192.168.77.2"
$BoardUser = "root"
$NetworkVideoPort = 5600
$TcpEventPort = 9000
$RtspPort = 8554
$RtspUrl = "rtsp://$BoardIp`:$RtspPort/live"
$CameraName = "Integrated Camera"
$FfmpegPath = "D:\EVCapture\ffmpeg.exe"

$BoardExecutable = "/root/edgevision_minimal/build-native-rtsp/edge_vision"
$ModelPath = "/root/edgevision_minimal/models/yolo11s_rk3568_i8.rknn"
$LabelsPath = "/root/edgevision_minimal/assets/coco_80_labels_list.txt"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $env:TEMP "edgevision-runtime"
$StatePath = Join-Path $RuntimeDir "launcher-state.json"
$EventLogPath = Join-Path $RuntimeDir "event_log.csv"
$ReadyTimeoutSeconds = 20

function Find-FirstExisting([string[]] $Candidates, [string] $Description) {
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    throw "$Description not found. Checked: $($Candidates -join ', ')"
}

function Test-Port([string] $TargetHost, [int] $Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($TargetHost, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(1000)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Wait-Port([string] $TargetHost, [int] $Port, [string] $Description) {
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $TargetHost $Port) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "$Description was not ready on $TargetHost`:$Port within $ReadyTimeoutSeconds seconds"
}

function Stop-ProcessTree([int] $ProcessId) {
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        & taskkill.exe /PID $ProcessId /T /F | Out-Null
    }
}

function Save-State([int] $BridgePid, [int] $EventLoggerPid) {
    [pscustomobject]@{
        bridge_pid = $BridgePid
        event_logger_pid = $EventLoggerPid
        board_ip = $BoardIp
        board_started = $true
        event_log = $EventLogPath
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Stop-BoardNormally {
    $stopCommand = "pkill -TERM -x edge_vision 2>/dev/null || true; sleep 1; echo EDGEVISION_STOP_REQUESTED"
    & ssh.exe -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$BoardUser@$BoardIp" $stopCommand 2>&1 | Out-Host
}

$bridge = $null
$eventLogger = $null
$boardStarted = $false
$startupComplete = $false
$startupFailed = $false

try {
    if (-not (Get-Command ssh.exe -ErrorAction SilentlyContinue)) {
        throw "[1/5] OpenSSH ssh.exe is not available"
    }
    $pcBridge = Find-FirstExisting @(
        (Join-Path $RepoRoot "build-pc-tools\Release\edgevision_pc_bridge.exe"),
        (Join-Path $RepoRoot "build-pc-bridge\Release\edgevision_pc_bridge.exe"),
        "D:\rk3568\pc-bridge-build\Release\edgevision_pc_bridge.exe"
    ) "PC bridge executable"
    $eventLoggerExecutable = Find-FirstExisting @(
        (Join-Path $RepoRoot "build-pc-tools\Release\edgevision_event_logger.exe"),
        (Join-Path $RepoRoot "build-pc-event-logger\Release\edgevision_event_logger.exe"),
        "D:\rk3568\edgevision_event_logger.exe"
    ) "PC event logger executable"
    if (-not (Test-Path -LiteralPath $FfmpegPath -PathType Leaf)) {
        throw "FFmpeg executable not found: $FfmpegPath"
    }
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

    $remoteStart = "if [ ! -x '$BoardExecutable' ]; then echo BOARD_EXECUTABLE_MISSING; exit 10; fi; if [ ! -f '$ModelPath' ]; then echo MODEL_MISSING; exit 11; fi; if [ ! -f '$LabelsPath' ]; then echo LABELS_MISSING; exit 12; fi; if pgrep -x edge_vision >/dev/null; then echo EDGEVISION_ALREADY_RUNNING; exit 2; fi; nohup env DISPLAY=:0 '$BoardExecutable' --model '$ModelPath' --labels '$LabelsPath' --input network --show --smooth-preview --tcp --tcp-port $TcpEventPort --conf 0.15 > /tmp/edgevision-launcher.log 2>&1 < /dev/null & echo EDGEVISION_STARTED"
    $remoteOutput = @(& ssh.exe -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$BoardUser@$BoardIp" $remoteStart 2>&1)
    if ($LASTEXITCODE -ne 0) {
        if ($remoteOutput -match "EDGEVISION_ALREADY_RUNNING") {
            throw "[2/5] EdgeVision is already running on the board; run tools/stop_edgevision.ps1 first"
        }
        throw "[1/5]/[2/5] Board SSH/start failed: $($remoteOutput -join ' ')"
    }
    if ($remoteOutput -notmatch "EDGEVISION_STARTED") {
        throw "[2/5] Board start was not confirmed: $($remoteOutput -join ' ')"
    }
    Write-Host "[1/5] RK3568 reachable"
    Write-Host "[2/5] EdgeVision started"
    $boardStarted = $true

    Wait-Port $BoardIp $TcpEventPort "TCP event server"
    Wait-Port $BoardIp $RtspPort "RTSP server"

    $bridgeStdout = Join-Path $RuntimeDir "pc-bridge.stdout.log"
    $bridgeStderr = Join-Path $RuntimeDir "pc-bridge.stderr.log"
    $bridgeArguments = "--ffmpeg `"$FfmpegPath`" --camera `"$CameraName`" --board $BoardIp --input-port $NetworkVideoPort --rtsp-port $RtspPort"
    $bridge = Start-Process -FilePath $pcBridge -WorkingDirectory $RuntimeDir -ArgumentList $bridgeArguments `
        -RedirectStandardOutput $bridgeStdout -RedirectStandardError $bridgeStderr -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    if ($bridge.HasExited) {
        throw "[3/5] PC camera sender/preview bridge exited early; see $bridgeStdout and $bridgeStderr"
    }
    Wait-Port $BoardIp $NetworkVideoPort "MJPEG/TCP input server"
    Write-Host "[3/5] Camera sender and Raw Preview started"
    Start-Sleep -Seconds 4
    if ($bridge.HasExited) {
        throw "[4/5] RTSP preview bridge exited early; see $bridgeStdout and $bridgeStderr"
    }
    Write-Host "[4/5] RTSP preview started: $RtspUrl"

    $eventLogStdout = Join-Path $RuntimeDir "event-logger.stdout.log"
    $eventLogStderr = Join-Path $RuntimeDir "event-logger.stderr.log"
    $eventLoggerArguments = "--host $BoardIp --port $TcpEventPort --output `"$EventLogPath`""
    $eventLogger = Start-Process -FilePath $eventLoggerExecutable -WorkingDirectory $RuntimeDir -ArgumentList $eventLoggerArguments `
        -RedirectStandardOutput $eventLogStdout -RedirectStandardError $eventLogStderr -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    if ($eventLogger.HasExited) {
        throw "[5/5] TCP event subscriber exited early; see $eventLogStdout"
    }
    Save-State $bridge.Id $eventLogger.Id
    Write-Host "[5/5] Event subscriber started: $EventLogPath"
    Write-Host "EdgeVision ready. Press Ctrl+C to stop PC helpers, then run tools/stop_edgevision.ps1 to stop the board normally."
    $startupComplete = $true

    while (-not $bridge.HasExited -and -not $eventLogger.HasExited) {
        Start-Sleep -Seconds 1
    }
    if ($bridge.HasExited) {
        Write-Warning "PC bridge exited; inspect $bridgeStdout and $bridgeStderr"
    }
    if ($eventLogger.HasExited) {
        Write-Warning "Event logger exited; inspect $eventLogStdout"
    }
}
catch {
    $startupFailed = $true
    Write-Error $_.Exception.Message
}
finally {
    if ($null -ne $eventLogger -and -not $eventLogger.HasExited) {
        Stop-Process -Id $eventLogger.Id -ErrorAction SilentlyContinue
    }
    if ($null -ne $bridge -and -not $bridge.HasExited) {
        Stop-ProcessTree $bridge.Id
    }
    if ($startupFailed -and $boardStarted -and -not $startupComplete) {
        Write-Warning "Stopping the board because startup did not complete"
        Stop-BoardNormally
    }
    if ($startupFailed) {
        exit 1
    }
}
