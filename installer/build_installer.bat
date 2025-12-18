@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Daemon Installer Build Script
echo ========================================
echo.

:: Check for Inno Setup Compiler
where iscc >nul 2>&1
if errorlevel 1 (
    :: Try default installation path
    if exist "C:\Program Files (x86)\Inno Setup 6\iscc.exe" (
        set "ISCC=C:\Program Files (x86)\Inno Setup 6\iscc.exe"
    ) else if exist "C:\Program Files\Inno Setup 6\iscc.exe" (
        set "ISCC=C:\Program Files\Inno Setup 6\iscc.exe"
    ) else (
        echo ERROR: Inno Setup Compiler ^(iscc^) not found
        echo.
        echo Please install Inno Setup 6 from:
        echo   https://jrsoftware.org/issetup.html
        echo.
        echo Or add it to your PATH:
        echo   C:\Program Files ^(x86^)\Inno Setup 6\
        echo.
        exit /b 1
    )
) else (
    set "ISCC=iscc"
)

:: Check for PyInstaller output
if not exist "..\dist\Daemon\Daemon.exe" (
    echo ERROR: PyInstaller output not found at ..\dist\Daemon\Daemon.exe
    echo.
    echo Please build the executable first:
    echo   cd ..
    echo   pyinstaller daemon.spec --clean --noconfirm
    echo.
    exit /b 1
)

:: Check for icon file
if not exist "..\assets\daemon_icon.ico" (
    echo WARNING: Icon file not found at ..\assets\daemon_icon.ico
    echo The installer may not have the correct icon.
    echo.
)

:: Check for license file
if not exist "LICENSE.txt" (
    echo ERROR: LICENSE.txt not found in installer directory
    echo.
    exit /b 1
)

:: Create output directory if it doesn't exist
if not exist "output" mkdir output

:: Build installer
echo Building installer...
echo Using: %ISCC%
echo.

"%ISCC%" daemon_installer.iss

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Installer build failed
    echo ========================================
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS: Installer created
echo ========================================
echo.
echo Output: installer\output\DaemonSetup-1.0.0.exe
echo.
echo Next steps:
echo   1. Test the installer on a clean Windows system
echo   2. Upload to your distribution platform
echo.

pause
