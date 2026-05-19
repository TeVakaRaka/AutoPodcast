@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo === AutoPodcast: Build ===

:: Find Python
set "PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=python"
    ) else (
        echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
        exit /b 1
    )
)

echo Using Python: %PYTHON%
%PYTHON% --version

:: Create venv
if not exist "_build\venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON% -m venv _build\venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        exit /b 1
    )
)

:: Activate venv
call _build\venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

:: Build
echo Building with PyInstaller...
pyinstaller build.spec --distpath _build/dist --workpath _build/build -y
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)

:: Copy ffmpeg
if exist "ffmpeg\ffmpeg.exe" (
    echo Copying ffmpeg...
    copy /Y "ffmpeg\ffmpeg.exe" "_build\dist\autopodcast\" >nul
    copy /Y "ffmpeg\ffprobe.exe" "_build\dist\autopodcast\" >nul
) else (
    echo WARNING: ffmpeg not found. Run DOWNLOAD_FFMPEG.bat first.
    echo The build will work but ffmpeg features will require ffmpeg in PATH.
)

:: Verify
if exist "_build\dist\autopodcast\autopodcast.exe" (
    echo.
    echo Build successful.
    echo Output: _build\dist\autopodcast\autopodcast.exe
) else (
    echo ERROR: autopodcast.exe not found after build.
    exit /b 1
)

endlocal
