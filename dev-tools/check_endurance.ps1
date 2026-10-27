param(
    [switch]$Cuda,
    [switch]$NativeCpu,
    [string]$ModelDirectory = '',
    [string]$Python = 'python',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$workspace = Split-Path -Parent $PSScriptRoot
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $workspace ('target/endurance-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
} elseif (![IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $workspace $OutputDirectory
}
if (Test-Path -LiteralPath $OutputDirectory) {throw "Use a new evidence directory: $OutputDirectory"}
if ($ModelDirectory) {
    $ModelDirectory = (Resolve-Path -LiteralPath $ModelDirectory).Path
    foreach ($name in @('model.safetensors', 'tokenizer.json')) {
        if (!(Test-Path -LiteralPath (Join-Path $ModelDirectory $name) -PathType Leaf)) {throw "Missing real-model asset: $name"}
    }
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$saved = @{}
foreach ($name in @('RUSTFLAGS', 'LUMEN_CUDA', 'PATH', 'LUMEN_LONG_MODEL_DIR')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $workspace
try {
    $env:PATH = (Get-LumenReleaseDirectory -Workspace $workspace) + [IO.Path]::PathSeparator + $env:PATH
    $env:LUMEN_CUDA = if ($Cuda) {'1'} else {'0'}
    if ($ModelDirectory) {$env:LUMEN_LONG_MODEL_DIR = $ModelDirectory}
    else {Remove-Item Env:LUMEN_LONG_MODEL_DIR -ErrorAction SilentlyContinue}
    $features = @('dev-tools')
    if ($Cuda) {$features += 'cuda'}
    if ($NativeCpu) {
        $env:RUSTFLAGS = '-C target-cpu=native'
        if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq 'X64') {
            $features += @('x86-fp-kernels', 'x86-int8-kernels')
        }
    } else {Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue}
    $build = @(& cargo test --release --locked --lib --no-run --message-format=json --no-default-features --features ($features -join ',') 2> (Join-Path $OutputDirectory 'build.log'))
    $code = $LASTEXITCODE
    $build | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory 'build.jsonl')
    if ($code -ne 0) {Get-Content (Join-Path $OutputDirectory 'build.log') -Tail 30; throw 'Endurance test build failed'}
    $artifacts = @($build | Where-Object {$_ -match '^\{'} | ForEach-Object {ConvertFrom-Json $_} | Where-Object {$_.reason -eq 'compiler-artifact' -and $_.profile.test -and $_.executable -and $_.target.name -eq 'lumen'})
    if ($artifacts.Count -ne 1) {throw 'Expected one library-test executable'}
    $exe = if ($env:OS -eq 'Windows_NT') {'endurance-tests.exe'} else {'endurance-tests'}
    $binary = Join-Path $OutputDirectory $exe
    Copy-Item -LiteralPath $artifacts[0].executable -Destination $binary
    $tests = @(& $binary --list --ignored | ForEach-Object {if ($_ -match '^(endurance_tests::[^ ]+): test$') {$Matches[1]}} | Sort-Object @{Expression = {if ($_ -match 'real_weight') {1} else {0}}}, @{Expression = {$_}})
    $expected = if ($Cuda) {8} else {4}
    if ($tests.Count -ne $expected) {throw "Expected $expected endurance tests, found $($tests.Count)"}
    $index = @()
    foreach ($test in $tests) {
        $tag = $test.Split(':')[-1]
        if ($tag -match 'real_weight' -and !$ModelDirectory) {
            $index += [pscustomobject]@{test = $test; status = 'skipped'; reason = 'Local model directory not supplied; not counted as passed'}
        } else {
            $watch = [Diagnostics.Stopwatch]::StartNew()
            & $binary --exact $test --ignored --nocapture --test-threads=1 *> (Join-Path $OutputDirectory "$tag.log")
            $code = $LASTEXITCODE
            $watch.Stop()
            $index += [pscustomobject]@{test = $test; status = $(if ($code -eq 0) {'passed'} else {'failed'}); returncode = $code; seconds = $watch.Elapsed.TotalSeconds; log = "$tag.log"}
            Write-Output "$tag : $($index[-1].status)"
        }
        ConvertTo-Json -Depth 6 -InputObject @($index) | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory 'run-index.json')
    }
    if (@($index | Where-Object {$_.status -eq 'failed'}).Count) {throw 'Endurance runtime check failed; all logs preserved'}
    $referenceArgs = @((Join-Path $PSScriptRoot 'check_endurance.py'), $OutputDirectory)
    if ($Cuda) {$referenceArgs += '--cuda'}
    if ($ModelDirectory) {$referenceArgs += @('--model-directory', $ModelDirectory)}
    & $Python @referenceArgs *> (Join-Path $OutputDirectory 'reference.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'reference.log') -Tail 30; throw 'Independent endurance reference failed'}
    Get-Content (Join-Path $OutputDirectory 'reference.log')
    [pscustomobject]@{binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant(); features = ($features -join ','); rustflags = $env:RUSTFLAGS; model_directory = $ModelDirectory} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory 'configuration.json')
} finally {
    Pop-Location
    foreach ($name in $saved.Keys) {[Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')}
}
