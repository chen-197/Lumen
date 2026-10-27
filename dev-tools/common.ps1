# Resolve Cargo's configured target directory, including CARGO_TARGET_DIR and
# .cargo/config.toml overrides, instead of assuming workspace/target/release.
function Get-LumenReleaseDirectory {
    param([Parameter(Mandatory = $true)][string]$Workspace)
    $manifest = Join-Path $Workspace 'Cargo.toml'
    $json = @(& cargo metadata --format-version 1 --no-deps --offline --manifest-path $manifest)
    if ($LASTEXITCODE -ne 0) {throw 'Cannot resolve the Cargo target directory'}
    $metadata = ($json -join "`n") | ConvertFrom-Json
    return Join-Path $metadata.target_directory 'release'
}
