@echo off
REM Quick start script for KTO Lunar Lander Demo (Windows)
REM Requires: Docker Desktop and VcXsrv installed

echo ============================================================
echo KTO Lunar Lander Demo - Docker Setup
echo ============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/4] Docker is running...

REM Check if VcXsrv is running (look for X11 on port 6000)
netstat -an | findstr ":6000" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: VcXsrv may not be running!
    echo.
    echo Please ensure VcXsrv is running with:
    echo   - Display: 0
    echo   - "Disable access control" checked
    echo.
    echo Press any key to continue anyway...
    pause >nul
) else (
    echo [2/4] VcXsrv is running...
)

echo [3/4] Building Docker image (this may take 5-10 minutes first time)...
docker-compose --profile windows build

if errorlevel 1 (
    echo.
    echo ERROR: Docker build failed!
    pause
    exit /b 1
)

echo [4/4] Starting container...
echo.
echo ============================================================
echo Controls:
echo   Arrow Up: Main engine (Teleop mode)
echo   Left/Right: Rotate
echo   R: Reset
echo   Q/Escape: Quit
echo   1: Teleop  2: Heuristic  3: KTO mode
echo ============================================================
echo.

docker-compose --profile windows up

echo.
echo Demo stopped.
pause
