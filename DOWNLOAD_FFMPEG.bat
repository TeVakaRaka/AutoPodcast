@echo off
chcp 65001 >nul 2>&1
setlocal

echo === AutoPodcast: Download ffmpeg ===

if exist "ffmpeg\ffmpeg.exe" (
    if exist "ffmpeg\ffprobe.exe" (
        echo ffmpeg already downloaded, skipping.
        goto :end
    )
)

set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "ZIP_FILE=_build\ffmpeg-download.zip"

if not exist "_build" mkdir "_build"
if not exist "ffmpeg" mkdir "ffmpeg"

echo Downloading ffmpeg from BtbN/FFmpeg-Builds...
curl -L -o "%ZIP_FILE%" "%FFMPEG_URL%"
if errorlevel 1 (
    echo ERROR: Failed to download ffmpeg.
    exit /b 1
)

echo Extracting ffmpeg.exe and ffprobe.exe...
tar -xf "%ZIP_FILE%" --strip-components=2 -C ffmpeg "ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe" "ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe"
if errorlevel 1 (
    echo ERROR: Failed to extract ffmpeg.
    exit /b 1
)

del "%ZIP_FILE%" 2>nul

if exist "ffmpeg\ffmpeg.exe" (
    echo ffmpeg downloaded successfully.
) else (
    echo ERROR: ffmpeg.exe not found after extraction.
    exit /b 1
)

:end
endlocal
