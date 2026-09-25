<#
.SYNOPSIS
  Native Windows-on-ARM64 build of trellis.cpp.

.DESCRIPTION
  ggml refuses MSVC for ARM CPU kernels (src/ggml-cpu/CMakeLists.txt:105), so this
  builds with clang using the GNU-style driver targeting arm64-pc-windows-msvc --
  the same approach upstream llama.cpp uses for Windows ARM64.

  By default ggml probes the local CPU and falls back to -mcpu=native on Windows.
  Pass -Arch to disable native detection and target an explicit, redistributable
  ARM architecture instead.

.EXAMPLE
  scripts\build-arm64.ps1 -Backend cpu
  scripts\build-arm64.ps1 -Backend vulkan
  scripts\build-arm64.ps1 -Backend hexagon -HexagonSdk C:\Qualcomm\Hexagon_SDK\6.4.0.0
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'vulkan', 'hexagon')]
    [string]$Backend = 'cpu',

    [string]$BuildDir = '',

    # Empty = let ggml probe the CPU itself. Its probe actually *runs* test programs
    # (check_cxx_source_runs), so it verifies features rather than trusting -mcpu=native,
    # which over-reports. Pass an explicit arch (e.g. 'armv8.7-a+dotprod+i8mm+fp16') to
    # pin it instead -- needed for redistributable builds.
    [string]$Arch = '',

    [string]$HexagonSdk = $env:HEXAGON_SDK_ROOT,

    [int]$Jobs = 0,

    [switch]$Clean,

    # Extra -D flags passed straight to cmake, e.g. -ExtraArgs '-DGGML_VULKAN_DEBUG=ON'
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

# A shell started before the Vulkan SDK was installed will not have VULKAN_SDK in its
# process environment, so fall back to the machine-level value and then to a scan.
function Resolve-VulkanSdk {
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) { return }
    $machine = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
    if ($machine -and (Test-Path $machine)) { $env:VULKAN_SDK = $machine; return }
    $found = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
             Sort-Object Name -Descending | Select-Object -First 1
    if ($found) { $env:VULKAN_SDK = $found.FullName; return }
    throw "Vulkan SDK not found. Install the Windows ARM64 SDK from https://sdk.lunarg.com/"
}

$repo = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $repo "build-arm64-$Backend" }
$repo = [IO.Path]::GetFullPath($repo).TrimEnd('\')
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }

# ---- locate Visual Studio ---------------------------------------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; is Visual Studio installed?" }
$vsroot = & $vswhere -latest -products * -property installationPath
if (-not $vsroot) { throw "no Visual Studio installation found" }

$vcvarsall = Join-Path $vsroot 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvarsall)) { throw "vcvarsall.bat not found under $vsroot" }

# ---- import the arm64 developer environment into this session ---------------
Write-Host "[build] importing arm64 developer environment" -ForegroundColor Cyan
$envDump = cmd /c "`"$vcvarsall`" arm64 >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat arm64 failed" }
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($Matches[1])" -Value $Matches[2] }
}

# VS ships cmake/ninja but does not put them on PATH.
$cmakeDir = Join-Path $vsroot 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$env:PATH = "$cmakeDir\CMake\bin;$cmakeDir\Ninja;$env:PATH"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not on PATH. Install the 'MSVC v14.x - C++ ARM64/ARM64EC build tools' component."
}

# ---- locate clang -----------------------------------------------------------
$clang = $null
foreach ($c in @(
        (Join-Path $vsroot 'VC\Tools\Llvm\ARM64\bin\clang.exe'),
        (Join-Path $vsroot 'VC\Tools\Llvm\bin\clang.exe'),
        (Join-Path $vsroot 'VC\Tools\Llvm\x64\bin\clang.exe'),
        'C:\Program Files\LLVM\bin\clang.exe')) {
    if (Test-Path $c) { $clang = $c; break }
}
if (-not $clang) {
    $onPath = Get-Command clang.exe -ErrorAction SilentlyContinue
    if ($onPath) { $clang = $onPath.Source }
}
if (-not $clang) {
    throw @"
clang not found. ggml cannot build its ARM CPU backend with MSVC.
Install either:
  * VS component  Microsoft.VisualStudio.Component.VC.Llvm.Clang, or
  * standalone LLVM for Windows-on-ARM  (winget install LLVM.LLVM)
"@
}
$clangxx = Join-Path (Split-Path $clang) 'clang++.exe'
if (-not (Test-Path $clangxx)) { $clangxx = $clang }
Write-Host "[build] clang: $clang" -ForegroundColor Cyan

# ---- backend-specific cmake arguments --------------------------------------
$cmakeArgs = @(
    '-B', $BuildDir,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_C_COMPILER=$($clang -replace '\\','/')",
    "-DCMAKE_CXX_COMPILER=$($clangxx -replace '\\','/')",
    '-DCMAKE_C_COMPILER_TARGET=arm64-pc-windows-msvc',
    '-DCMAKE_CXX_COMPILER_TARGET=arm64-pc-windows-msvc',
    '-DGGML_OPENMP=OFF'
)

if ($Arch) {
    $cmakeArgs += '-DGGML_NATIVE=OFF'
    $cmakeArgs += "-DGGML_CPU_ARM_ARCH=$Arch"
} else {
    # ggml will warn "ARM -march/-mcpu not found" because its probe reads /dev/null,
    # which does not exist on Windows. Harmless: it falls back to -mcpu=native and the
    # per-feature checks that follow are real runtime tests.
    $cmakeArgs += '-DGGML_NATIVE=ON'
}

switch ($Backend) {
    'cpu' { }
    'vulkan' {
        Resolve-VulkanSdk
        $cmakeArgs += '-DGGML_VULKAN=ON'
    }
    'hexagon' {
        if (-not $HexagonSdk) {
            throw "Hexagon SDK path not given. Pass -HexagonSdk <path> or set HEXAGON_SDK_ROOT."
        }
        if (-not (Test-Path $HexagonSdk)) { throw "Hexagon SDK not found at: $HexagonSdk" }

        # The trimmed Windows-on-Snapdragon SDK records the compiler path in
        # hexagon_sdk.json. The SDK's top-level CMake also expects PREBUILT_LIB_DIR
        # before ggml configures its per-DSP external projects.
        $sdkConfig = Join-Path $HexagonSdk 'hexagon_sdk.json'
        if (-not (Test-Path $sdkConfig)) { throw "hexagon_sdk.json not found under: $HexagonSdk" }
        $sdkMetadata = Get-Content -LiteralPath $sdkConfig -Raw | ConvertFrom-Json
        $toolsInfo = $sdkMetadata.root.tools.info |
                     Where-Object name -EQ 'Hexagon Tools' |
                     Select-Object -First 1
        if (-not $toolsInfo.path) { throw "Hexagon Tools path not found in: $sdkConfig" }
        $hexagonTools = Join-Path $HexagonSdk $toolsInfo.path
        if (-not (Test-Path $hexagonTools)) { throw "Hexagon Tools not found at: $hexagonTools" }

        # Vulkan alongside Hexagon: ops the NPU cannot take fall back to Adreno
        # rather than all the way to the CPU.
        if ($env:VULKAN_SDK) { $cmakeArgs += '-DGGML_VULKAN=ON' }
        $cmakeArgs += '-DGGML_HEXAGON=ON'
        $cmakeArgs += "-DHEXAGON_SDK_ROOT=$($HexagonSdk -replace '\\','/')"
        $cmakeArgs += "-DHEXAGON_TOOLS_ROOT=$($hexagonTools -replace '\\','/')"
        $cmakeArgs += '-DPREBUILT_LIB_DIR=toolv19_v81'
    }
}

$cmakeArgs += $ExtraArgs

if ($Clean -and (Test-Path $BuildDir)) {
    $insideRepo = $BuildDir.StartsWith($repo + [IO.Path]::DirectorySeparatorChar,
                                       [StringComparison]::OrdinalIgnoreCase)
    $buildLeaf = Split-Path -Leaf $BuildDir
    if (-not $insideRepo -or -not $buildLeaf.StartsWith('build-')) {
        throw "refusing to clean unexpected build directory: $BuildDir"
    }
    Write-Host "[build] removing $BuildDir" -ForegroundColor Yellow
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

Push-Location $repo
# cmake and ninja write progress and warnings to stderr. Under $ErrorActionPreference =
# 'Stop' PowerShell turns any native stderr line into a terminating NativeCommandError,
# which aborts the build on a harmless warning. Judge these by exit code instead.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    Write-Host "[build] configure ($Backend) -> $BuildDir" -ForegroundColor Cyan
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

    Write-Host "[build] compiling with $Jobs jobs" -ForegroundColor Cyan
    & cmake --build $BuildDir -j $Jobs
    if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

    Write-Host "[build] done -> $BuildDir" -ForegroundColor Green
} finally {
    $ErrorActionPreference = $prevEAP
    Pop-Location
}
