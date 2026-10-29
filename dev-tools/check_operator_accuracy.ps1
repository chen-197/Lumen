param(
    [switch]$Cuda,
    [switch]$NativeCpu,
    [switch]$SkipFormatCheck,
    [string]$Python = 'python',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $workspace ('target/operator-accuracy-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
} elseif (![IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $workspace $OutputDirectory
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$saved = @{}
foreach ($name in @('RUSTFLAGS', 'LUMEN_CUDA', 'LUMEN_OPERATOR_AUDIT')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location $workspace
try {
    $env:LUMEN_CUDA = if ($Cuda) {'1'} else {'0'}
    $env:LUMEN_OPERATOR_AUDIT = '1'
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
    $testLog = Join-Path $OutputDirectory 'tests.log'
    cargo test --release --locked --all-targets --no-default-features --features ($features -join ',') -- --nocapture --test-threads=1 *> $testLog
    if ($LASTEXITCODE -ne 0) {Get-Content $testLog -Tail 45; throw 'Rust accuracy/regression tests failed'}
    cargo test --release --locked --doc --no-default-features --features ($features -join ',') *> (Join-Path $OutputDirectory 'doctests.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'doctests.log') -Tail 35; throw 'Public API documentation tests failed'}
    $referenceArgs = @((Join-Path $PSScriptRoot 'check_operator_accuracy.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'torch-comparison.json'))
    if ($Cuda) {$referenceArgs += '--cuda'}
    & $Python @referenceArgs *> (Join-Path $OutputDirectory 'reference.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference.log') -Tail 35
        throw 'Independent PyTorch reference failed (Python must have torch installed)'
    }
    $denseArgs = @((Join-Path $PSScriptRoot 'check_dense_cross_entropy.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'dense-ce-comparison.json'))
    if ($Cuda) {$denseArgs += '--cuda'}
    & $Python @denseArgs *> (Join-Path $OutputDirectory 'reference-dense-ce.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-dense-ce.log') -Tail 35
        throw 'Independent dense cross entropy reference failed'
    }
    $reuseArgs = @((Join-Path $PSScriptRoot 'check_autograd_reuse.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'autograd-reuse-comparison.json'))
    if ($Cuda) {$reuseArgs += '--cuda'}
    & $Python @reuseArgs *> (Join-Path $OutputDirectory 'reference-autograd-reuse.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-autograd-reuse.log') -Tail 35
        throw 'Independent retained-graph backward reference failed'
    }
    $flagsArgs = @((Join-Path $PSScriptRoot 'check_gradient_flags.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'gradient-flags-comparison.json'))
    if ($Cuda) {$flagsArgs += '--cuda'}
    & $Python @flagsArgs *> (Join-Path $OutputDirectory 'reference-gradient-flags.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-gradient-flags.log') -Tail 35
        throw 'Independent frozen-input and QKV gradient reference failed'
    }
    $accumArgs = @((Join-Path $PSScriptRoot 'check_gradient_accumulation.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'gradient-accumulation-comparison.json'))
    if ($Cuda) {$accumArgs += '--cuda'}
    & $Python @accumArgs *> (Join-Path $OutputDirectory 'reference-gradient-accumulation.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-gradient-accumulation.log') -Tail 35
        throw 'Independent mixed-residency gradient accumulation reference failed'
    }
    $genericArgs = @((Join-Path $PSScriptRoot 'check_generic_nn.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'generic-nn-comparison.json'))
    if ($Cuda) {$genericArgs += '--cuda'}
    & $Python @genericArgs *> (Join-Path $OutputDirectory 'reference-generic-nn.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-generic-nn.log') -Tail 35
        throw 'Independent general attention/LayerNorm reference failed'
    }
    $prefillArgs = @((Join-Path $PSScriptRoot 'check_generic_prefill.py'), $testLog,
        '--output', (Join-Path $OutputDirectory 'generic-prefill-comparison.json'))
    & $Python @prefillArgs *> (Join-Path $OutputDirectory 'reference-generic-prefill.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-generic-prefill.log') -Tail 35
        throw 'Independent general CPU prefill reference failed'
    }
    & $Python (Join-Path $PSScriptRoot 'check_row_matmul.py') $testLog --output (Join-Path $OutputDirectory 'row-matmul-comparison.json') *> (Join-Path $OutputDirectory 'reference-row-matmul.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-row-matmul.log') -Tail 35
        throw 'Independent mixed-precision row matmul reference failed'
    }
    & $Python (Join-Path $PSScriptRoot 'check_qwen3.py') $testLog --output (Join-Path $OutputDirectory 'qwen3-comparison.json') *> (Join-Path $OutputDirectory 'reference-qwen3.log')
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $OutputDirectory 'reference-qwen3.log') -Tail 30
        throw 'Independent Qwen3 forward/gradient reference failed'
    }
    cargo clippy --release --locked --all-targets --no-default-features --features ($features -join ',') -- -D warnings -A clippy::chunks_exact_to_as_chunks *> (Join-Path $OutputDirectory 'clippy.log')
    if ($LASTEXITCODE -ne 0) {Get-Content (Join-Path $OutputDirectory 'clippy.log') -Tail 35; throw 'Clippy failed'}
    if (!$SkipFormatCheck) {
        cargo fmt --all --check *> (Join-Path $OutputDirectory 'fmt.log')
        if ($LASTEXITCODE -ne 0) {throw 'Formatting check failed'}
    }
    Get-Content (Join-Path $OutputDirectory 'reference.log')
    Get-Content (Join-Path $OutputDirectory 'reference-dense-ce.log')
    Get-Content (Join-Path $OutputDirectory 'reference-autograd-reuse.log')
    Get-Content (Join-Path $OutputDirectory 'reference-gradient-flags.log')
    Get-Content (Join-Path $OutputDirectory 'reference-gradient-accumulation.log')
    Get-Content (Join-Path $OutputDirectory 'reference-generic-nn.log')
    Get-Content (Join-Path $OutputDirectory 'reference-generic-prefill.log')
    Get-Content (Join-Path $OutputDirectory 'reference-row-matmul.log')
    Get-Content (Join-Path $OutputDirectory 'reference-qwen3.log')
    Write-Output "Operator accuracy gate passed. Evidence: $OutputDirectory"
} finally {
    Pop-Location
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
