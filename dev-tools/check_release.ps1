param(
    [switch]$Cuda,
    [switch]$NativeCpu,
    [switch]$Endurance,
    [string]$ModelDirectory = '',
    [string]$Python = 'python',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$workspace = Split-Path -Parent $PSScriptRoot
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $workspace ('target/release-check-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
} elseif (![IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $workspace $OutputDirectory
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$saved = @{}
foreach ($name in @('RUSTFLAGS', 'RUSTDOCFLAGS', 'LUMEN_CUDA', 'PATH')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $workspace
try {
    $env:PATH = (Get-LumenReleaseDirectory -Workspace $workspace) + [IO.Path]::PathSeparator + $env:PATH
    & (Join-Path $PSScriptRoot 'check_operator_accuracy.ps1') -Python $Python -OutputDirectory (Join-Path $OutputDirectory 'portable')
    if ($LASTEXITCODE -ne 0) {throw 'Portable accuracy checks failed'}
    if ($Cuda -or $NativeCpu) {
        $nativeArgs = @{NativeCpu = $true; SkipFormatCheck = $true; Python = $Python; OutputDirectory = (Join-Path $OutputDirectory 'native')}
        if ($Cuda) {$nativeArgs.Cuda = $true}
        & (Join-Path $PSScriptRoot 'check_operator_accuracy.ps1') @nativeArgs
        if ($LASTEXITCODE -ne 0) {throw 'Native accuracy checks failed'}
    }
    $manualArgs = @{Python = $Python; OutputDirectory = (Join-Path $OutputDirectory 'manual')}
    if ($Cuda) {$manualArgs.Cuda = $true}
    if ($Cuda -or $NativeCpu) {$manualArgs.NativeCpu = $true}
    & (Join-Path $PSScriptRoot 'check_manual_state.ps1') @manualArgs
    if ($LASTEXITCODE -ne 0) {throw 'Public API manual state checks failed'}
    if ($Endurance) {
        $enduranceArgs = @{Python = $Python; OutputDirectory = (Join-Path $OutputDirectory 'endurance'); ModelDirectory = $ModelDirectory}
        if ($Cuda) {$enduranceArgs.Cuda = $true}
        if ($Cuda -or $NativeCpu) {$enduranceArgs.NativeCpu = $true}
        & (Join-Path $PSScriptRoot 'check_endurance.ps1') @enduranceArgs
    }
    $env:LUMEN_CUDA = '0'
    Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    $env:RUSTDOCFLAGS = ($env:RUSTDOCFLAGS + ' -D warnings').Trim()
    cargo doc --locked --no-default-features --no-deps *> (Join-Path $OutputDirectory 'doc.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'doc.log') -Tail 30; throw 'Documentation build failed'}
    cargo package --allow-dirty --locked --list *> (Join-Path $OutputDirectory 'package-files.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'package-files.log') -Tail 30; throw 'Package listing failed'}
    cargo package --allow-dirty --locked *> (Join-Path $OutputDirectory 'package.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'package.log') -Tail 40; throw 'Local package verification failed'}
    git diff --check *> (Join-Path $OutputDirectory 'diff-check.log')
    if ($LASTEXITCODE -ne 0) {throw 'Diff check failed'}
    Write-Output "Local release checks passed. No publication was attempted. Evidence: $OutputDirectory"
} finally {
    Pop-Location
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
