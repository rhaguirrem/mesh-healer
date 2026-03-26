#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Builds OpenMeshCraft (Arrangements + CDT) and fTetWild from source on Windows.

.DESCRIPTION
    1. Installs prerequisites via winget (Git, CMake, VS 2022 Build Tools)
    2. Clones and builds OpenMeshCraft -> OpenMeshCraft-Arrangements.exe, OpenMeshCraft-CDT.exe
    3. Clones and builds fTetWild       -> FloatTetwild_bin.exe
    4. Copies binaries to $ToolsDir and sets persistent environment variables.

.PARAMETER ToolsDir
    Directory where final executables are placed. Default: C:\Tools\MeshHealBackends

.PARAMETER SourceDir
    Directory for cloned source repos. Default: C:\Projects\_build_backends

.PARAMETER SkipPrereqs
    Skip winget prerequisite installation (if you already have Git, CMake, VS).

.PARAMETER BoostRoot
    Path to a pre-installed Boost library root (containing include/ and lib/).
    If not set, the script will install Boost via vcpkg.
#>
param(
    [string]$ToolsDir  = "C:\Tools\MeshHealBackends",
    [string]$SourceDir = "C:\Projects\_build_backends",
    [switch]$SkipPrereqs,
    [string]$BoostRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n========== $msg ==========" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# 0. Refresh PATH so newly-installed tools are discoverable
# ---------------------------------------------------------------------------
function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path    = "$machinePath;$userPath"
}

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
Write-Step "1/6  Checking & installing prerequisites"

if (-not $SkipPrereqs) {
    # Git
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "  Installing Git ..."
        winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
        Refresh-Path
    }
    if (Get-Command git -ErrorAction SilentlyContinue) { Write-Ok "Git $(git --version)" } else { throw "Git installation failed." }

    # CMake
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Host "  Installing CMake ..."
        winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements
        Refresh-Path
    }
    if (Get-Command cmake -ErrorAction SilentlyContinue) { Write-Ok "CMake $(cmake --version | Select-Object -First 1)" } else { throw "CMake installation failed." }

    # Visual Studio 2022 Build Tools (C++ workload)
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $hasVS = $false
    if (Test-Path $vsWhere) {
        $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsInstall) { $hasVS = $true }
    }
    if (-not $hasVS) {
        Write-Host "  Installing Visual Studio 2022 Build Tools (C++ workload) ..."
        Write-Host "  This is a large download (~2-6 GB). Please be patient." -ForegroundColor Yellow
        winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
            --accept-source-agreements --accept-package-agreements `
            --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        Refresh-Path
    }
    # Re-check
    if (Test-Path $vsWhere) {
        $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsInstall) { Write-Ok "VS Build Tools at $vsInstall" } else { throw "VS Build Tools C++ workload not found after install." }
    } else {
        throw "vswhere not found. VS Build Tools installation may have failed."
    }
} else {
    Write-Warn "Skipping prerequisite installation (--SkipPrereqs)"
    if (-not (Get-Command git   -ErrorAction SilentlyContinue)) { throw "git not on PATH" }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw "cmake not on PATH" }
}

# ---------------------------------------------------------------------------
# 2. Prepare directories
# ---------------------------------------------------------------------------
Write-Step "2/6  Preparing directories"
New-Item -ItemType Directory -Force -Path $ToolsDir  | Out-Null
New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
Write-Ok "Tools  -> $ToolsDir"
Write-Ok "Source -> $SourceDir"

# ---------------------------------------------------------------------------
# Helper: find the VS developer environment and import it
# ---------------------------------------------------------------------------
function Import-VsDevEnv {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

    Write-Host "  Importing VS dev environment from $vcvars ..."
    $output = cmd /c "`"$vcvars`" >nul 2>&1 && set" 2>&1
    foreach ($line in $output) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
    Write-Ok "cl.exe version: $((Get-Command cl -ErrorAction SilentlyContinue).Source)"
}

Import-VsDevEnv

# ---------------------------------------------------------------------------
# 3. Install vcpkg & Boost (if no BoostRoot provided)
# ---------------------------------------------------------------------------
Write-Step "3/6  Setting up vcpkg & dependencies"
$vcpkgDir = Join-Path $SourceDir "vcpkg"
if (-not (Test-Path (Join-Path $vcpkgDir "vcpkg.exe"))) {
    Write-Host "  Cloning vcpkg ..."
    git clone https://github.com/microsoft/vcpkg.git $vcpkgDir
    & (Join-Path $vcpkgDir "bootstrap-vcpkg.bat") -disableMetrics
}
$vcpkgExe = Join-Path $vcpkgDir "vcpkg.exe"
$env:VCPKG_ROOT = $vcpkgDir
$vcpkgToolchain = Join-Path $vcpkgDir "scripts\buildsystems\vcpkg.cmake"
Write-Ok "vcpkg at $vcpkgExe"

# Install Boost (headers + filesystem + system) via vcpkg if needed
if (-not $BoostRoot) {
    Write-Host "  Installing Boost via vcpkg (this may take a while) ..."
    & $vcpkgExe install boost-filesystem:x64-windows boost-system:x64-windows boost-container-hash:x64-windows
    Write-Ok "Boost installed via vcpkg"
}

# Install GMP/MPFR/MPIR via vcpkg (needed by both projects)
Write-Host "  Installing GMP, MPFR, MPIR via vcpkg ..."
& $vcpkgExe install gmp:x64-windows mpfr:x64-windows mpir:x64-windows
Write-Ok "GMP/MPFR/MPIR installed"

# Install Eigen3, CGAL, TBB via vcpkg (needed by OpenMeshCraft)
Write-Host "  Installing Eigen3, CGAL, TBB via vcpkg ..."
& $vcpkgExe install eigen3:x64-windows cgal:x64-windows tbb:x64-windows
Write-Ok "Eigen3/CGAL/TBB installed"

# ---------------------------------------------------------------------------
# 4. Build OpenMeshCraft
# ---------------------------------------------------------------------------
Write-Step "4/6  Building OpenMeshCraft"
$omcSrc = Join-Path $SourceDir "OpenMeshCraft"
if (-not (Test-Path $omcSrc)) {
    git clone --recursive https://github.com/mangoleaves/OpenMeshCraft.git $omcSrc
} else {
    Write-Warn "OpenMeshCraft already cloned, pulling latest ..."
    Push-Location $omcSrc; git pull; Pop-Location
}

$omcBuild = Join-Path $omcSrc "build"
New-Item -ItemType Directory -Force -Path $omcBuild | Out-Null

$cmakeArgs = @(
    "-S", $omcSrc,
    "-B", $omcBuild,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
)
if ($BoostRoot) {
    $cmakeArgs += "-DBOOST_ROOT=$BoostRoot"
}

Write-Host "  Configuring OpenMeshCraft ..."
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "OpenMeshCraft CMake configuration failed." }

Write-Host "  Building OpenMeshCraft-Arrangements ..."
cmake --build $omcBuild --config Release --target OpenMeshCraft-Arrangements
if ($LASTEXITCODE -ne 0) { throw "OpenMeshCraft-Arrangements build failed." }

Write-Host "  Building OpenMeshCraft-CDT ..."
cmake --build $omcBuild --config Release --target OpenMeshCraft-CDT
if ($LASTEXITCODE -ne 0) { throw "OpenMeshCraft-CDT build failed." }

# Find and copy the built executables
$arrExe = Get-ChildItem -Path $omcBuild -Recurse -Filter "OpenMeshCraft-Arrangements.exe" | Select-Object -First 1
$cdtExe = Get-ChildItem -Path $omcBuild -Recurse -Filter "OpenMeshCraft-CDT.exe" | Select-Object -First 1

if (-not $arrExe) { throw "OpenMeshCraft-Arrangements.exe not found in build output." }
if (-not $cdtExe) { throw "OpenMeshCraft-CDT.exe not found in build output." }

Copy-Item $arrExe.FullName -Destination $ToolsDir -Force
Copy-Item $cdtExe.FullName -Destination $ToolsDir -Force
Write-Ok "OpenMeshCraft-Arrangements.exe -> $ToolsDir"
Write-Ok "OpenMeshCraft-CDT.exe -> $ToolsDir"

# ---------------------------------------------------------------------------
# 5. Build fTetWild
# ---------------------------------------------------------------------------
Write-Step "5/6  Building fTetWild"
$ftetSrc = Join-Path $SourceDir "fTetWild"
if (-not (Test-Path $ftetSrc)) {
    git clone --recursive https://github.com/wildmeshing/fTetWild.git $ftetSrc
} else {
    Write-Warn "fTetWild already cloned, pulling latest ..."
    Push-Location $ftetSrc; git pull; Pop-Location
}

$ftetBuild = Join-Path $ftetSrc "build"
New-Item -ItemType Directory -Force -Path $ftetBuild | Out-Null

Write-Host "  Configuring fTetWild ..."
cmake -S $ftetSrc -B $ftetBuild -G "Ninja" `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
if ($LASTEXITCODE -ne 0) { throw "fTetWild CMake configuration failed." }

Write-Host "  Building FloatTetwild_bin ..."
cmake --build $ftetBuild --config Release
if ($LASTEXITCODE -ne 0) { throw "fTetWild build failed." }

$ftetExe = Get-ChildItem -Path $ftetBuild -Recurse -Filter "FloatTetwild_bin.exe" | Select-Object -First 1
if (-not $ftetExe) { throw "FloatTetwild_bin.exe not found in build output." }

Copy-Item $ftetExe.FullName -Destination $ToolsDir -Force
Write-Ok "FloatTetwild_bin.exe -> $ToolsDir"

# Copy any required DLLs (mpir.dll etc.) next to the executable
$dlls = Get-ChildItem -Path $ftetBuild -Recurse -Filter "*.dll" -ErrorAction SilentlyContinue
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName -Destination $ToolsDir -Force
    Write-Ok "  DLL: $($dll.Name) -> $ToolsDir"
}
# Also copy vcpkg runtime DLLs
$vcpkgBinDir = Join-Path $vcpkgDir "installed\x64-windows\bin"
if (Test-Path $vcpkgBinDir) {
    foreach ($pattern in @("mpir*.dll", "gmp*.dll", "mpfr*.dll", "tbb*.dll")) {
        Get-ChildItem -Path $vcpkgBinDir -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $ToolsDir -Force
            Write-Ok "  vcpkg DLL: $($_.Name) -> $ToolsDir"
        }
    }
}

# ---------------------------------------------------------------------------
# 6. Set environment variables (User scope, persistent)
# ---------------------------------------------------------------------------
Write-Step "6/6  Setting environment variables"

$arrExePath = Join-Path $ToolsDir "OpenMeshCraft-Arrangements.exe"
$cdtExePath = Join-Path $ToolsDir "OpenMeshCraft-CDT.exe"
$ftetExePath = Join-Path $ToolsDir "FloatTetwild_bin.exe"

[Environment]::SetEnvironmentVariable("OPENMESHCRAFT_ARRANGEMENTS_EXE", $arrExePath, "User")
[Environment]::SetEnvironmentVariable("OPENMESHCRAFT_CDT_EXE",          $cdtExePath,  "User")
[Environment]::SetEnvironmentVariable("FASTTETWILD_EXE",                $ftetExePath, "User")

# Also add ToolsDir to user PATH if not already there
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$ToolsDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$ToolsDir", "User")
    Write-Ok "Added $ToolsDir to user PATH"
}

# Set for current session too
$env:OPENMESHCRAFT_ARRANGEMENTS_EXE = $arrExePath
$env:OPENMESHCRAFT_CDT_EXE          = $cdtExePath
$env:FASTTETWILD_EXE                = $ftetExePath
$env:Path                          += ";$ToolsDir"

Write-Ok "OPENMESHCRAFT_ARRANGEMENTS_EXE = $arrExePath"
Write-Ok "OPENMESHCRAFT_CDT_EXE          = $cdtExePath"
Write-Ok "FASTTETWILD_EXE                = $ftetExePath"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host "`n" -NoNewline
Write-Step "BUILD COMPLETE"
Write-Host @"

  All executables are in: $ToolsDir
  Environment variables have been set (User scope).
  Restart your terminal/IDE to pick up the new PATH.

  To verify:
    OpenMeshCraft-Arrangements.exe --help
    OpenMeshCraft-CDT.exe --help
    FloatTetwild_bin.exe --help

"@ -ForegroundColor Green
