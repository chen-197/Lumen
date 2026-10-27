param(
    [switch]$Cuda,
    [switch]$NativeCpu,
    [string]$Python = 'python',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$workspace = Split-Path -Parent $PSScriptRoot
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $workspace ('target/manual-state-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
} elseif (![IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $workspace $OutputDirectory
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$saved = @{}
foreach ($name in @('RUSTFLAGS', 'LUMEN_CUDA', 'PATH')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $workspace
try {
    $releaseDirectory = Get-LumenReleaseDirectory -Workspace $workspace
    $env:PATH = $releaseDirectory + [IO.Path]::PathSeparator + $env:PATH
    $env:LUMEN_CUDA = if ($Cuda) {'1'} else {'0'}
    $features = @('dev-tools')
    if ($Cuda) {$features += 'cuda'}
    if ($NativeCpu) {
        $env:RUSTFLAGS = '-C target-cpu=native'
        if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq 'X64') {
            $features += @('x86-fp-kernels', 'x86-int8-kernels')
        }
    } else {
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    }
    cargo build --release --locked --no-default-features --features ($features -join ',') --bin manual_state_check *> (Join-Path $OutputDirectory 'build.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'build.log') -Tail 30; throw 'Manual check build failed'}
    $exe = if ($env:OS -eq 'Windows_NT') {'manual_state_check.exe'} else {'manual_state_check'}
    $binary = Join-Path $OutputDirectory $exe
    Copy-Item -LiteralPath (Join-Path $releaseDirectory $exe) -Destination $binary
    $devices = @('cpu')
    if ($Cuda) {$devices += 'cuda'}
    foreach ($device in $devices) {
        $caseDirectory = Join-Path $OutputDirectory $device
        & $binary --device $device --output-directory $caseDirectory *> (Join-Path $OutputDirectory "$device.log")
        if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory "$device.log") -Tail 20; throw "Manual $device state checks failed"}
        & $Python (Join-Path $PSScriptRoot 'check_manual_state.py') (Join-Path $caseDirectory 'results.json') --output (Join-Path $OutputDirectory "$device-reference.json") *> (Join-Path $OutputDirectory "$device-reference.log")
        if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory "$device-reference.log") -Tail 25; throw "Independent $device state reference failed"}
        Get-Content (Join-Path $OutputDirectory "$device-reference.log")
    }
} finally {
    Pop-Location
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
