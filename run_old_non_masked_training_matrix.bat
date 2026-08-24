@echo off
setlocal

rem Usage: run_old_non_masked_training_matrix.bat [device] [batch_size] [num_workers]
rem Add --dry-run first to print the 42 runs without starting training.
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
)

set "DEVICE=%~1"
set "BATCH_SIZE=%~2"
set "NUM_WORKERS=%~3"
if not defined DEVICE set "DEVICE=auto"
if not defined BATCH_SIZE set "BATCH_SIZE=64"
if not defined NUM_WORKERS set "NUM_WORKERS=0"

set "START_DIR=%CD%"
cd /d "%~dp0" || exit /b 1

set "PYTHON=.venv\Scripts\python.exe"
set "TRAINER=pm25_transformer.py"
set "PAIRS=data\legacy\purpleair_continental_us_pairs_thinned_20km.csv"
set "INDOOR=..\purple-air-pull\purpleair_hourly_pm25_atm"
set "OUTDOOR=..\purple-air-pull\tempo_pm25_sensor_match\tempo_pm25_indoor_sensors.csv"
set "FORECASTS=naqfc_output"

for %%P in ("%PYTHON%" "%TRAINER%" "%PAIRS%" "%INDOOR%" "%OUTDOOR%" "%FORECASTS%") do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

set "MODEL_COUNT=0"
for /f "delims=" %%M in ('%PYTHON% -c "from pm25_models import model_names; print(*model_names(), sep='\n')"') do (
    set /a MODEL_COUNT+=1 >nul
    call :run_model "%%M" || goto :fail
)
if "%MODEL_COUNT%"=="0" (
    echo Could not read the available model names.
    goto :fail
)

cd /d "%START_DIR%"
exit /b 0

:run_model
call :run_one "%~1" 5 || exit /b 1
call :run_one "%~1" 20 || exit /b 1
call :run_one "%~1" 100 || exit /b 1
exit /b 0

:run_one
set "MODEL=%~1"
set "EPOCHS=%~2"
set "CHECKPOINT=old-non-masked-%MODEL%-%EPOCHS%ep.pt"
echo Starting %MODEL%: epochs=%EPOCHS% patience=%EPOCHS% checkpoint=%CHECKPOINT%
if defined DRY_RUN exit /b 0

%PYTHON% %TRAINER% train ^
    --model "%MODEL%" ^
    --pairs "%PAIRS%" ^
    --indoor-history "%INDOOR%" ^
    --outdoor-history "%OUTDOOR%" ^
    --forecast-root "%FORECASTS%" ^
    --epochs %EPOCHS% ^
    --early-stopping-patience %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --num-workers %NUM_WORKERS% ^
    --device "%DEVICE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Training failed for %MODEL% at %EPOCHS% epochs.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
