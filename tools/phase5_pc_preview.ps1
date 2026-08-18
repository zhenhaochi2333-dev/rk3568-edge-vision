param(
    [string]$FfmpegPath = "",
    [string]$FfplayPath = "",
    [string]$CameraName = "Integrated Camera",
    [int]$DurationSeconds = 0
)

$ErrorActionPreference = "Stop"

function Resolve-Executable([string]$requested, [string]$name) {
    if ($requested -ne "") {
        if (-not (Test-Path -LiteralPath $requested)) {
            throw "Executable not found: $requested"
        }
        return (Resolve-Path -LiteralPath $requested).Path
    }
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Unable to find $name; pass -$name`Path explicitly"
    }
    return $command.Source
}

$ffmpeg = Resolve-Executable $FfmpegPath "ffmpeg"
$ffplay = Resolve-Executable $FfplayPath "ffplay"

$ffmpegArguments = @(
    "-hide_banner",
    "-loglevel", "info",
    "-f", "dshow",
    "-video_size", "1280x720",
    "-framerate", "30",
    "-pixel_format", "nv12",
    "-i", "video=$CameraName",
    "-filter_complex", "[0:v]split=2[raw][enc]",
    "-map", "[raw]",
    "-pix_fmt", "yuv420p",
    "-c:v", "rawvideo",
    "-f", "nut",
    "pipe:1",
    "-map", "[enc]",
    "-fps_mode", "passthrough",
    "-c:v", "h264_mf",
    "-rate_control", "cbr",
    "-scenario", "live_streaming",
    "-hw_encoding", "true",
    "-g", "15",
    "-bf", "0",
    "-f", "rtp",
    "-payload_type", "96",
    "-pkt_size", "1200",
    "rtp://192.168.77.2:5600"
)

$ffplayArguments = @(
    "-hide_banner",
    "-loglevel", "info",
    "-stats",
    "-window_title", "EdgeVision Raw Camera",
    "-f", "nut",
    "-i", "-"
)

$ffplayInfo = [System.Diagnostics.ProcessStartInfo]::new()
$ffplayInfo.FileName = $ffplay
$ffplayInfo.UseShellExecute = $false
$ffplayInfo.RedirectStandardInput = $true
$ffplayInfo.RedirectStandardError = $false
$ffplayInfo.CreateNoWindow = $false
foreach ($argument in $ffplayArguments) {
    [void]$ffplayInfo.ArgumentList.Add($argument)
}

$ffmpegInfo = [System.Diagnostics.ProcessStartInfo]::new()
$ffmpegInfo.FileName = $ffmpeg
$ffmpegInfo.UseShellExecute = $false
$ffmpegInfo.RedirectStandardOutput = $true
$ffmpegInfo.RedirectStandardError = $true
$ffmpegInfo.CreateNoWindow = $true
foreach ($argument in $ffmpegArguments) {
    [void]$ffmpegInfo.ArgumentList.Add($argument)
}

$preview = [System.Diagnostics.Process]::new()
$preview.StartInfo = $ffplayInfo
$sender = [System.Diagnostics.Process]::new()
$sender.StartInfo = $ffmpegInfo

$copyTask = $null
$stderrTask = $null
try {
    if (-not $preview.Start()) {
        throw "Unable to start ffplay"
    }
    if (-not $sender.Start()) {
        throw "Unable to start ffmpeg"
    }

    $copyTask = $sender.StandardOutput.BaseStream.CopyToAsync(
        $preview.StandardInput.BaseStream)
    $stderrTask = $sender.StandardError.ReadToEndAsync()

    $deadline = if ($DurationSeconds -gt 0) {
        [DateTime]::UtcNow.AddSeconds($DurationSeconds)
    } else {
        [DateTime]::MaxValue
    }
    while (-not $sender.HasExited -and -not $preview.HasExited -and
           [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 200
    }
} finally {
    if ($sender -and -not $sender.HasExited) {
        $sender.Kill()
    }
    if ($preview -and -not $preview.HasExited) {
        $preview.Kill()
    }
    if ($copyTask) {
        try { $copyTask.GetAwaiter().GetResult() } catch { }
    }
    if ($stderrTask) {
        try { $stderrTask.GetAwaiter().GetResult() } catch { }
    }
    if ($sender) { $sender.Dispose() }
    if ($preview) { $preview.Dispose() }
}
