@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo [%date% %time%] START > autopodcast.log
echo Лог: autopodcast.log
echo.

echo ============================================
echo   AutoPodcast — auto-multicam
echo   .prproj + 2 микрофона = готовый проект
echo ============================================
echo.

:: ---- Проверка Python ----
set STEP=Проверка Python
echo [1/6] %STEP%...
python --version >nul 2>&1
if errorlevel 1 (
    echo   Python не найден в PATH
    echo   Скачай: https://www.python.org/downloads/
    echo   При установке отметь "Add Python to PATH"
    goto :error
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Python %PYVER%
echo.

:: ---- Проверка ffmpeg ----
set STEP=Проверка ffmpeg
echo [2/6] %STEP%...
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo   ffmpeg найден в PATH
    goto :ffmpeg_ok
)
:: Проверяем локальную копию
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" (
    echo   ffmpeg найден в tools\ffmpeg\bin\
    goto :ffmpeg_add_path
)
:: Скачиваем ffmpeg
echo   ffmpeg не найден, скачиваю...
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "FFMPEG_ZIP=%~dp0tools\ffmpeg_download.zip"
set "FFMPEG_TMP=%~dp0tools\ffmpeg_tmp"
if not exist "%~dp0tools" mkdir "%~dp0tools"
curl.exe -L -o "%FFMPEG_ZIP%" "%FFMPEG_URL%" 2>> autopodcast.log
if errorlevel 1 (
    echo   curl не сработал, пробую PowerShell...
    powershell -NoProfile -Command "(New-Object Net.WebClient).DownloadFile('%FFMPEG_URL%', '%FFMPEG_ZIP%')" >> autopodcast.log 2>&1
    if errorlevel 1 (
        echo   Не удалось скачать ffmpeg
        goto :error
    )
)
echo   Распаковываю...
powershell -NoProfile -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_TMP%' -Force; $sub = Get-ChildItem '%FFMPEG_TMP%' -Directory | Select-Object -First 1; if ($sub -and (Test-Path (Join-Path $sub.FullName 'bin\ffmpeg.exe'))) { Copy-Item (Join-Path $sub.FullName 'bin') -Destination '%~dp0tools\ffmpeg\bin' -Recurse -Force } else { Write-Error 'ffmpeg.exe not found in archive'; exit 1 }" >> autopodcast.log 2>&1
if errorlevel 1 (
    echo   Не удалось распаковать ffmpeg
    goto :error
)
:: Удаляем временные файлы
del "%FFMPEG_ZIP%" >nul 2>&1
rmdir /s /q "%FFMPEG_TMP%" >nul 2>&1
if not exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" (
    echo   Не удалось найти ffmpeg.exe после распаковки
    goto :error
)
echo   ffmpeg установлен в tools\ffmpeg\bin\
:ffmpeg_add_path
set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo   ffmpeg не найден после установки
    goto :error
)
:ffmpeg_ok
echo   ffmpeg OK
echo.

:: ---- Создание venv ----
set STEP=Создание venv
echo [3/6] %STEP%...
if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
    if errorlevel 1 goto :error
    echo   .venv создан
) else (
    echo   .venv уже существует
)
call .venv\Scripts\activate.bat
echo.

:: ---- Установка зависимостей ----
set STEP=Установка зависимостей
echo [4/6] %STEP%...
pip install -r requirements.txt -q >> autopodcast.log 2>&1
if errorlevel 1 goto :error
pip install -e . -q >> autopodcast.log 2>&1
if errorlevel 1 goto :error
echo   Зависимости установлены
echo.

:: ---- Тесты ----
set STEP=Тесты
echo [5/6] Запуск тестов...
python -m pytest tests/ -v >> autopodcast.log 2>&1
if errorlevel 1 (
    echo.
    echo *** ТЕСТЫ НЕ ПРОШЛИ ***
    echo Подробности в autopodcast.log
    goto :error
)
echo   Тесты пройдены
echo.

:: ---- Ввод данных ----
set STEP=Ввод данных
echo ============================================
echo [6/6] Настройка auto-multicam
echo ============================================
echo.

echo Перетащи .prproj файл из Premiere:
set /p "PRPROJ_FILE=.prproj файл: "
set "PRPROJ_FILE=!PRPROJ_FILE:"=!"
if not exist "!PRPROJ_FILE!" (
    echo   Файл не найден: !PRPROJ_FILE!
    goto :error
)
echo.

echo Перетащи WAV-файл микрофона ВЕДУЩЕГО (host):
set /p "MIC_A=Микрофон ведущего: "
set "MIC_A=!MIC_A:"=!"
if not exist "!MIC_A!" (
    echo   Файл не найден: !MIC_A!
    goto :error
)
echo.

echo Перетащи WAV-файл микрофона ГОСТЯ (guest):
set /p "MIC_B=Микрофон гостя: "
set "MIC_B=!MIC_B:"=!"
if not exist "!MIC_B!" (
    echo   Файл не найден: !MIC_B!
    goto :error
)
echo.

set /p "SEQ_NAME=Имя секвенции в проекте: "
if "!SEQ_NAME!"=="" (
    echo   Имя секвенции не может быть пустым
    goto :error
)
echo.

:: Опциональные параметры
set "EXTRA_OPTS="
echo Дополнительные настройки (Enter = по умолчанию):
echo.
set /p "CAM_HOST=Angle ведущего (default 1): "
if not "!CAM_HOST!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --camera-host !CAM_HOST!"
set /p "CAM_GUEST=Angle гостя (default 2): "
if not "!CAM_GUEST!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --camera-guest !CAM_GUEST!"
set /p "CAM_WIDE=Angle широкого плана (default 3): "
if not "!CAM_WIDE!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --camera-wide !CAM_WIDE!"
set /p "AUDIO_HOST=Аудиодорожка ведущего (default 1): "
if not "!AUDIO_HOST!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --audio-track-host !AUDIO_HOST!"
set /p "AUDIO_GUEST=Аудиодорожка гостя (default 2): "
if not "!AUDIO_GUEST!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --audio-track-guest !AUDIO_GUEST!"
set /p "MUTE=Мьютить неактивные микрофоны? (Y/n, default Y): "
if /i "!MUTE!"=="n" set "EXTRA_OPTS=!EXTRA_OPTS! --no-mute-audio"
set /p "THRESHOLD=Порог речи в dB (default -24.0): "
if not "!THRESHOLD!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --speech-threshold !THRESHOLD!"
set /p "GAIN=Входное усиление в dB (default 0.0): "
if not "!GAIN!"=="" set "EXTRA_OPTS=!EXTRA_OPTS! --input-gain !GAIN!"
echo.

:: Определяем выходной файл рядом с исходным
for %%F in ("!PRPROJ_FILE!") do (
    set "OUT_DIR=%%~dpF"
    set "OUT_NAME=%%~nF_multicam%%~xF"
)
set "OUT_FILE=!OUT_DIR!!OUT_NAME!"

:: ---- Запуск auto-multicam ----
set STEP=auto-multicam
echo Запуск анализа и патчинга...
echo.
python -m autopodcast auto-multicam ^
    --in "!PRPROJ_FILE!" ^
    --mic-a "!MIC_A!" ^
    --mic-b "!MIC_B!" ^
    --seq "!SEQ_NAME!" ^
    --out "!OUT_FILE!" ^
    !EXTRA_OPTS!
if errorlevel 1 goto :error

echo.
echo ============================================
echo   ГОТОВО
echo.
echo   Результат: !OUT_FILE!
echo.
echo   Открой в Premiere Pro:
echo     File ^> Open Project ^> !OUT_NAME!
echo ============================================
pause
exit /b 0

:error
echo.
echo *** ОШИБКА на шаге: %STEP% ***
echo Подробности в autopodcast.log
echo [%date% %time%] ОШИБКА на шаге: %STEP% >> autopodcast.log
pause
exit /b 1
