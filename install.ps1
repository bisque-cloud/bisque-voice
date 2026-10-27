# Install `bisque-voice` on Windows - the local voice engine behind /present.
#
#   irm https://download.bisque.today/bisque-voice/install.ps1 | iex
#
# Installs to %USERPROFILE%\.bisque\bin. The POSIX twin is install.sh; the two
# resolve the same latest.json and verify the same checksums, so a fix to one
# belongs in the other.
#
# Written against Windows PowerShell 5.1 - the version that ships in the box -
# using nothing that needs PowerShell 7 or an installed package manager. The
# person running this is trying to get a voice, not a toolchain.
#
# Overridable by environment (they work through `irm | iex`, where parameters
# cannot be passed) or by parameter when the file is run directly:
#
#   $env:BISQUE_VOICE_BASE_URL   release root, laid out as latest.json and
#                              v{VERSION}/. Defaults to this repository's
#                              GitHub Releases. A file:// URL or a local
#                              directory works, which is how the download ->
#                              verify -> extract path is exercised offline.
#                              Versions before 0.1.38 were not built on
#                              GitHub; set it to
#                              https://download.bisque.today/bisque-voice to
#                              install one of those.
#   $env:BISQUE_BIN_DIR        install directory
#   $env:BISQUE_VOICE_VERSION    pin a version instead of resolving latest.json
#
# Keep this file pure ASCII. Windows PowerShell 5.1 reads a BOM-less file as
# the legacy ANSI codepage, so an em dash here reaches the user as mojibake;
# and adding a BOM instead would put a stray U+FEFF in front of the script
# text that `irm | iex` evaluates.

# Write-Host is the correct cmdlet for an installer: this is console output for
# a human, not pipeline data a caller might capture.
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '')]
param(
  [string]$BaseUrl = $(if ($env:BISQUE_VOICE_BASE_URL) { $env:BISQUE_VOICE_BASE_URL } else { 'https://github.com/bisque-cloud/bisque-voice/releases/download' }),
  [string]$BinDir = $(if ($env:BISQUE_BIN_DIR) { $env:BISQUE_BIN_DIR } else { Join-Path ([Environment]::GetFolderPath('UserProfile')) '.bisque\bin' }),
  [string]$Version = $env:BISQUE_VOICE_VERSION
)

# Any failing cmdlet must stop the install, not print red and carry on to
# extract a half-downloaded archive.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar makes Invoke-WebRequest ~10x slower on 5.1

function Fail([string]$Message) {
  Write-Host "bisque-voice: $Message" -ForegroundColor Red
  exit 1
}

# 5.1 is the floor because that is what Windows ships; anything older predates
# Invoke-WebRequest -UseBasicParsing behaving sanely and Get-FileHash existing.
if ($PSVersionTable.PSVersion.Major -lt 5) {
  Fail "PowerShell 5.1 or newer is required (found $($PSVersionTable.PSVersion))."
}

# On Windows this is always set; its absence means this script is being run on
# a non-Windows PowerShell, where install.sh is the right answer.
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if (-not $arch) {
  Fail "this does not look like Windows. Install with:`n  curl -fsSL https://download.bisque.today/bisque-voice/install.sh | sh"
}

switch ($arch.ToUpperInvariant()) {
  'AMD64' { $target = 'x86_64-pc-windows-msvc' }
  'ARM64' {
    # There is no aarch64-pc-windows-msvc build: ONNX Runtime's prebuilt
    # Windows-on-ARM package is not what `ort` fetches here. Windows 11 on ARM
    # emulates x64 well enough for this to work, and saying so up front beats
    # someone discovering it from a slow synthesis.
    $target = 'x86_64-pc-windows-msvc'
    Write-Host "note: no native ARM64 build exists - installing the x86_64 build, which runs under Windows' x64 emulation." -ForegroundColor Yellow
  }
  'X86' {
    Fail "32-bit Windows is not supported - the models do not fit in a 32-bit address space. A 64-bit Windows install is required."
  }
  default { Fail "unsupported processor architecture '$arch'." }
}

# tar has shipped in Windows 10 since 1803 (bsdtar). Rather than reimplement
# gzip+tar in PowerShell, require it and say exactly what is missing.
if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
  Fail "'tar' was not found. It ships with Windows 10 1803 and newer - on an older build, extract $BaseUrl/vX.Y.Z/bisque-voice-$target.tar.gz by hand."
}

# Windows PowerShell 5.1 negotiates whatever SecurityProtocol was configured
# machine-wide, which on untouched installs can still exclude TLS 1.2 - and
# then every HTTPS fetch here fails with an unhelpful "connection closed".
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
  Write-Verbose "SecurityProtocol is managed by the runtime here: $($_.Exception.Message)"
}

# One fetch path for http(s), file:// and plain directories. The local cases
# are not a convenience - they are how this script is testable at all before a
# release exists to point it at.
function Get-Payload {
  param([Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination)

  if ($Url -match '^https?://') {
    Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
  } else {
    $path = if ($Url -match '^file:') { ([Uri]$Url).LocalPath } else { $Url }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      throw "no such file: $path"
    }
    Copy-Item -LiteralPath $path -Destination $Destination -Force
  }
}

# GitHub serves the newest release's files under a fixed path of their own, not
# beside the versioned directories, so the default resolves latest.json there.
$latestUrl = if ($env:BISQUE_VOICE_BASE_URL -or $PSBoundParameters.ContainsKey('BaseUrl')) { "$BaseUrl/latest.json" } else { 'https://github.com/bisque-cloud/bisque-voice/releases/latest/download/latest.json' }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("bisque-voice-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

try {
  if (-not $Version) {
    $manifestPath = Join-Path $tmp 'latest.json'
    try {
      Get-Payload -Url $latestUrl -Destination $manifestPath
    } catch {
      Fail "could not fetch $latestUrl - $($_.Exception.Message)"
    }
    # -Raw because 5.1's ConvertFrom-Json chokes on a string[] from Get-Content.
    $Version = (Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json).version
    if (-not $Version) { Fail "could not read a version out of $latestUrl" }
  }
  $Version = $Version.TrimStart('v')

  $archiveName = "bisque-voice-$target.tar.gz"
  $archiveUrl = "$BaseUrl/v$Version/$archiveName"
  $archivePath = Join-Path $tmp $archiveName

  Write-Host "bisque-voice $Version ($target)"

  try {
    Get-Payload -Url $archiveUrl -Destination $archivePath
  } catch {
    Fail "download failed: $archiveUrl`n  $($_.Exception.Message)"
  }

  # Verify BEFORE extracting, and treat a missing checksum as a failure rather
  # than a reason to skip the check. Every release publishes one; if it is
  # absent, something is wrong with the download and running the binary anyway
  # is the exact habit this project refuses to teach.
  $sumPath = "$archivePath.sha256"
  try {
    Get-Payload -Url "$archiveUrl.sha256" -Destination $sumPath
  } catch {
    Fail "no checksum published at $archiveUrl.sha256 - refusing to install an unverified binary."
  }

  # `shasum -a 256` writes "<hash>  <filename>"; take the hash.
  $expected = ((Get-Content -Raw -LiteralPath $sumPath).Trim() -split '\s+')[0]
  $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
  if (-not $expected) { Fail "the published checksum file was empty - refusing to install." }
  if ($expected -ne $actual) {
    Write-Host ""
    Write-Host "  expected $expected" -ForegroundColor Red
    Write-Host "  actual   $actual" -ForegroundColor Red
    Fail "CHECKSUM MISMATCH for $archiveName - refusing to install. Retry; if it persists, do not run the file."
  }

  # Read the member list first: a surprise here means nothing has been written
  # into the install directory yet.
  # Trimmed because bsdtar's line endings are not worth trusting across hosts,
  # and a stray CR would turn the membership check below into a false alarm.
  $members = @(& tar -tzf $archivePath | ForEach-Object { $_.Trim() } | Where-Object { $_ })
  if ($LASTEXITCODE -ne 0) { Fail "the archive could not be read (tar exit $LASTEXITCODE)." }
  if ($members -notcontains 'bisque-voice.exe') {
    Fail "$archiveName does not contain bisque-voice.exe - got: $($members -join ', ')"
  }

  if (-not (Test-Path -LiteralPath $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
  }

  # Extract EVERY member. The .exe is not self-contained - ort's Windows
  # runtime ships DirectML.dll (and may ship more) beside it, and Windows
  # resolves those from the executable's own directory. Installing only the
  # .exe produces something that cannot start.
  & tar -xzf $archivePath -C $BinDir
  if ($LASTEXITCODE -ne 0) {
    Fail "extraction failed (tar exit $LASTEXITCODE). If bisque-voice is running, close it and re-run - Windows will not let a running .exe be replaced."
  }

  $exe = Join-Path $BinDir 'bisque-voice.exe'
  if (-not (Test-Path -LiteralPath $exe)) { Fail "extraction finished but $exe is missing." }

  Write-Host "installed: $exe"
  $extras = @($members | Where-Object { $_ -ne 'bisque-voice.exe' })
  if ($extras.Count -gt 0) {
    Write-Host "           plus $($extras -join ', ') (required - Windows loads them from this directory)"
  }
} finally {
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

# Print the PATH change; never make it. Rewriting someone's environment
# without asking is how an installer earns a reputation.
$onPath = ($env:PATH -split ';' | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') }) -contains $BinDir.TrimEnd('\')
if (-not $onPath) {
  Write-Host ""
  Write-Host "add it to your PATH (this session):"
  Write-Host "  `$env:PATH = `"$BinDir;`$env:PATH`""
  Write-Host "and permanently, for your user:"
  # Built from the stored User value, not $env:PATH - `setx` would flatten the
  # machine PATH into the user one and truncate it at 1024 characters.
  Write-Host "  [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';$BinDir', 'User')"
}

# An upgrade keeps the models already on disk, so only point at a download when
# the new binary reports none it can run. Same check as install.sh: `engines`
# labels an engine `[installed]` only when this build can open it.
$engines = try { & $exe engines 2>$null | Out-String } catch { '' }
if ($engines -notmatch '\[installed\]') {
  Write-Host ""
  Write-Host "no voice model is installed yet - pick one:"
  Write-Host "  bisque-voice engines          # the catalogue, with sizes and trade-offs"
  Write-Host "  bisque-voice install kokoro   # small, fast, English"
}
