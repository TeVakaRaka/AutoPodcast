@echo off
chcp 65001 >nul 2>&1
echo Running AutoPodcast tests...
echo.

python -m pytest tests/ -v
if %errorlevel% neq 0 (
    echo.
    echo Tests FAILED
    pause
    exit /b 1
)

echo.
echo All tests passed
pause
