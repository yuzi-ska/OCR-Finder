@echo off
setlocal EnableExtensions
set "OCR_FINDER_SKIP_PAUSE="
if defined OCR_FINDER_NO_PAUSE set "OCR_FINDER_SKIP_PAUSE=1"
if defined CI set "OCR_FINDER_SKIP_PAUSE=1"
if not defined OCR_FINDER_SKIP_PAUSE for /f %%I in ('powershell -NoProfile -Command "try { if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected -or [Console]::IsErrorRedirected) { Write-Output 1 } } catch { }"') do set "OCR_FINDER_SKIP_PAUSE=%%I"

echo ============================================================
echo OCR Finder - CPU Virtual Environment Setup
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or higher from https://www.python.org
    call :maybe_pause
    exit /b 1
)

echo [INFO] Python version:
python --version
echo.

REM Create virtual environment
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        call :maybe_pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    call :maybe_pause
    exit /b 1
)

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] pip upgrade failed, continuing with current version...
)
echo.

REM Install RapidOCR for CPU (lighter weight than PaddleOCR)
echo [INFO] Installing RapidOCR ONNX Runtime for CPU...
pip install rapidocr_onnxruntime
if errorlevel 1 (
    echo [ERROR] Failed to install RapidOCR
    call :maybe_pause
    exit /b 1
)
echo [SUCCESS] RapidOCR installed.
echo.

REM Install minimal dependencies for GUI and CPU limiting
echo [INFO] Installing additional dependencies...
pip install Pillow>=10.0.0 psutil>=5.9.0 nuitka>=2.6
if errorlevel 1 (
    echo [WARNING] Some dependencies failed to install
)
echo.

echo [SUCCESS] All CPU dependencies installed!
echo.

REM Display success message
echo ============================================================
echo CPU Setup Complete!
echo ============================================================
echo.
echo CPU version uses RapidOCR ONNX Runtime for lightweight OCR.
echo RapidOCR has built-in models - no additional model caching needed.
echo.
echo To run the program:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Run GUI: python ocr_finder_gui.py
echo   3. Or CLI: python ocr_finder.py -t "你好"
echo.
echo To build the offline CPU package:
echo   run: build_cpu.bat
echo.
call :maybe_pause
endlocal
goto :eof

:maybe_pause
if defined OCR_FINDER_SKIP_PAUSE goto :eof
pause
goto :eof
