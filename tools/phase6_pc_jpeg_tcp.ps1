param(
    [string]$FfmpegPath = "",
    [string]$FfplayPath = "",
    [string]$CameraName = "Integrated Camera",
    [int]$DurationSeconds = 0,
    [switch]$NoRawPreview
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
        throw "Unable to find $name; pass -${name}Path explicitly"
    }
    return $command.Source
}

$ffmpeg = Resolve-Executable $FfmpegPath "ffmpeg"
$ffplay = $null
if ($FfplayPath -ne "") {
    $ffplay = Resolve-Executable $FfplayPath "ffplay"
} else {
    $ffplayCommand = Get-Command ffplay -ErrorAction SilentlyContinue
    if ($null -ne $ffplayCommand) {
        $ffplay = $ffplayCommand.Source
    }
}

# One dshow input is split locally: raw preview and 15 FPS high-quality JPEG.
# q:v 3 is the fixed, validated quality for the TCP path; it is not swept.
$rawOutputArguments = if ($NoRawPreview) {
    @("-f", "null", "NUL")
} else {
    @("-f", "nut", "pipe:1")
}
$ffmpegArguments = @(
    "-hide_banner",
    "-loglevel", "info",
    "-f", "dshow",
    "-video_size", "1280x720",
    "-framerate", "30",
    "-pixel_format", "nv12",
    "-i", "video=$CameraName",
    "-filter_complex", "[0:v]split=2[raw][jpeg0];[jpeg0]fps=15[jpeg]",
    "-map", "[raw]",
    "-pix_fmt", "yuv420p",
    "-c:v", "rawvideo"
) + $rawOutputArguments + @(
    "-map", "[jpeg]",
    "-q:v", "3",
    "-c:v", "mjpeg",
    "-an",
    "-f", "mjpeg",
    "tcp://192.168.77.2:5600?tcp_nodelay=1"
)

$previewArguments = if ($null -ne $ffplay) {
    @(
        "-hide_banner",
        "-loglevel", "info",
        "-stats",
        "-window_title", "EdgeVision Raw Camera",
        "-f", "nut",
        "-i", "-"
    )
} else {
    # Some Windows FFmpeg bundles omit ffplay but include SDL output. Keep
    # the single-camera raw preview in that case using the same FFmpeg build.
    @(
        "-hide_banner",
        "-loglevel", "error",
        "-f", "nut",
        "-i", "-",
        "-f", "sdl",
        "EdgeVision Raw Camera"
    )
}

$ffplayInfo = [System.Diagnostics.ProcessStartInfo]::new()
$ffplayInfo.FileName = if ($null -ne $ffplay) { $ffplay } else { $ffmpeg }
$ffplayInfo.UseShellExecute = $false
$ffplayInfo.RedirectStandardInput = $true
$ffplayInfo.RedirectStandardError = $false
$ffplayInfo.CreateNoWindow = $false
foreach ($argument in $previewArguments) {
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

$preview = $null
if (-not $NoRawPreview) {
    $preview = [System.Diagnostics.Process]::new()
    $preview.StartInfo = $ffplayInfo
}
$sender = [System.Diagnostics.Process]::new()
$sender.StartInfo = $ffmpegInfo

$copyTask = $null
$stderrTask = $null
try {
    if (-not $NoRawPreview) {
        if (-not $preview.Start()) {
            throw "Unable to start raw preview"
        }
    }
    if (-not $sender.Start()) {
        throw "Unable to start ffmpeg"
    }

    if (-not $NoRawPreview) {
        $copyTask = $sender.StandardOutput.BaseStream.CopyToAsync(
            $preview.StandardInput.BaseStream)
    }
    $stderrTask = $sender.StandardError.ReadToEndAsync()

    $deadline = if ($DurationSeconds -gt 0) {
        [DateTime]::UtcNow.AddSeconds($DurationSeconds)
    } else {
        [DateTime]::MaxValue
    }
    while ((-not $sender.HasExited) -and
           ($NoRawPreview -or (-not $preview.HasExited)) -and
           ([DateTime]::UtcNow -lt $deadline)) {
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
