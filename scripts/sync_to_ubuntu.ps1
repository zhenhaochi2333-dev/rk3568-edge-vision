param(
    [string]$UbuntuHost = "topeet@192.168.254.128",
    [string]$RemoteRoot = "/home/topeet/edgevision/rk3568-edge-vision"
)

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
& ssh $UbuntuHost "mkdir -p '$RemoteRoot'"
if ($LASTEXITCODE -ne 0) { throw "Unable to create Ubuntu source directory" }

# The Windows tree is authoritative. Generated build/deploy/output products
# stay local and are intentionally excluded from this source sync.
& scp -r `
    (Join-Path $sourceRoot "CMakeLists.txt"),
    (Join-Path $sourceRoot "include"),
    (Join-Path $sourceRoot "src"),
    (Join-Path $sourceRoot "cmake"),
    (Join-Path $sourceRoot "scripts"),
    (Join-Path $sourceRoot "docs"),
    (Join-Path $sourceRoot "tests"),
    (Join-Path $sourceRoot "assets"),
    (Join-Path $sourceRoot "models"),
    (Join-Path $sourceRoot "deps\rknn_runtime_1.6.0\include"),
    "${UbuntuHost}:$RemoteRoot/"
if ($LASTEXITCODE -ne 0) { throw "Source sync failed" }
