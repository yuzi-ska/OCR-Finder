@echo off
setlocal EnableExtensions
set "OCR_FINDER_SKIP_PAUSE="
if defined OCR_FINDER_NO_PAUSE set "OCR_FINDER_SKIP_PAUSE=1"
if defined CI set "OCR_FINDER_SKIP_PAUSE=1"
if not defined OCR_FINDER_SKIP_PAUSE for /f %%I in ('powershell -NoProfile -Command "try { if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected -or [Console]::IsErrorRedirected) { Write-Output 1 } } catch { }"') do set "OCR_FINDER_SKIP_PAUSE=%%I"

set "OUTPUT_DIR=release"
set "APP_NAME=OCRFinder"
set "PACKAGE_DIR="
set "PACKAGE_SIZE_BYTES="
set "MAX_PACKAGE_BYTES=209715200"

echo ============================================================
echo OCR Finder - Build Offline CPU Package (RapidOCR)
echo ============================================================
echo.

if not exist venv (
    echo [ERROR] Virtual environment not found!
    echo Please run setup_cpu_venv.bat first to create the environment.
    call :maybe_pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    call :maybe_pause
    exit /b 1
)

echo [INFO] Verifying RapidOCR is installed...
python -c "from rapidocr_onnxruntime import RapidOCR; print('[INFO] RapidOCR is available')" 2>nul
if errorlevel 1 (
    echo [ERROR] RapidOCR is not installed. Run setup_cpu_venv.bat first.
    call :maybe_pause
    exit /b 1
)

echo [INFO] Building CPU package with RapidOCR (lightweight ONNX runtime)...
echo.

set "ENTRY_SCRIPT=ocr_finder_gui.py"

REM Check if Nuitka is installed
echo [INFO] Checking Nuitka...
python -c "import nuitka" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing Nuitka...
    pip install nuitka
    if errorlevel 1 (
        echo [ERROR] Failed to install Nuitka
        call :maybe_pause
        exit /b 1
    )
)

REM Clean previous builds
echo [INFO] Cleaning previous builds...
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"

REM Build standalone package (no PaddleOCR models needed - RapidOCR has built-in ONNX models)
echo [INFO] Building GUI package with Nuitka...
echo [INFO] This may take several minutes, especially on the first build.
echo.

python -m nuitka ^
    --standalone ^
    --mingw64 ^
    --assume-yes-for-downloads ^
    --enable-plugin=tk-inter ^
    --windows-console-mode=disable ^
    --noinclude-unittest-mode=nofollow ^
    --noinclude-dll=opencv_videoio_ffmpeg*.dll ^
    --output-dir="%OUTPUT_DIR%" ^
    --output-filename="%APP_NAME%.exe" ^
    --include-package=rapidocr_onnxruntime ^
    --include-package-data=rapidocr_onnxruntime ^
    --include-package=psutil ^
    "%ENTRY_SCRIPT%"
if errorlevel 1 (
    echo [ERROR] Build failed!
    call :maybe_pause
    exit /b 1
)

REM Remove unnecessary video processing DLL (not needed for OCR)
if exist "%OUTPUT_DIR%\ocr_finder_gui.dist\cv2\opencv_videoio_ffmpeg*.dll" del /q "%OUTPUT_DIR%\ocr_finder_gui.dist\cv2\opencv_videoio_ffmpeg*.dll" 2>nul
REM Also remove from root if present
if exist "%OUTPUT_DIR%\ocr_finder_gui.dist\opencv_videoio_ffmpeg*.dll" del /q "%OUTPUT_DIR%\ocr_finder_gui.dist\opencv_videoio_ffmpeg*.dll" 2>nul

for /f "usebackq delims=" %%D in (`python -c "from pathlib import Path; matches = sorted(Path(r'%OUTPUT_DIR%').glob('*.dist')); print(matches[0].resolve() if matches else '')"`) do set "PACKAGE_DIR=%%D"
if not defined PACKAGE_DIR (
    echo [ERROR] CPU build completed but no Nuitka .dist folder was found under %OUTPUT_DIR%.
    call :maybe_pause
    exit /b 1
)

for /d %%D in ("%OUTPUT_DIR%\*.build") do if exist "%%~fD" rmdir /s /q "%%~fD"

if exist LICENSE copy LICENSE "%PACKAGE_DIR%\" >nul 2>&1

for /f "usebackq delims=" %%S in (`python -c "from pathlib import Path; package_dir = Path(r'%PACKAGE_DIR%'); total = sum(path.stat().st_size for path in package_dir.rglob('*') if path.is_file()); print(total)"`) do set "PACKAGE_SIZE_BYTES=%%S"
if not defined PACKAGE_SIZE_BYTES (
    echo [ERROR] Failed to calculate CPU package size.
    call :maybe_pause
    exit /b 1
)

echo.
echo [SUCCESS] Build completed!
echo.

echo ============================================================
echo Build Summary
echo ============================================================
echo.
echo Package folder: %PACKAGE_DIR%
echo Executable: %PACKAGE_DIR%\%APP_NAME%.exe
echo.
echo CPU package uses RapidOCR ONNX Runtime (lightweight OCR backend).
echo RapidOCR has built-in ONNX models - no additional model bundling needed.
echo.
echo Package size: %PACKAGE_SIZE_BYTES% bytes
if %PACKAGE_SIZE_BYTES% GTR %MAX_PACKAGE_BYTES% (
    echo [WARNING] CPU package exceeds the 200MB limit.
    echo Consider if the size is acceptable for distribution.
) else (
    echo [SUCCESS] CPU package size is within the 200MB limit.
)
echo.
call :maybe_pause
endlocal
goto :eof

:maybe_pause
if defined OCR_FINDER_SKIP_PAUSE goto :eof
pause
goto :eof
