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
set "LINEAR_DATA=inputs\old_training_data.csv"
set "CYCLICAL_DATA=inputs\old_training_data_cyclical.csv"
set "LINEAR_CACHE=inference\keller_elementary_school_cache.pt"
set "CYCLICAL_CACHE=inference\keller_elementary_school_cache_cyclical.pt"
set "INFERENCE_ROOT=inference\old_non_masked_matrix"

for %%P in ("%PYTHON%" "%TRAINER%" "%LINEAR_DATA%" "%CYCLICAL_DATA%" "%LINEAR_CACHE%" "%CYCLICAL_CACHE%") do (
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
set "TRAINING_DATA=%LINEAR_DATA%"
set "CACHE=%LINEAR_CACHE%"
if not "%MODEL:cyclical=%"=="%MODEL%" (
    set "TRAINING_DATA=%CYCLICAL_DATA%"
    set "CACHE=%CYCLICAL_CACHE%"
)
set "INFERENCE_DIR=%INFERENCE_ROOT%\%MODEL%\%EPOCHS%ep"
echo Starting %MODEL%: epochs=%EPOCHS% patience=%EPOCHS% data=%TRAINING_DATA% checkpoint=%CHECKPOINT%
echo Inference examples: cache=%CACHE% output=%INFERENCE_DIR%
if defined DRY_RUN exit /b 0

%PYTHON% %TRAINER% train ^
    --model "%MODEL%" ^
    --training-data "%TRAINING_DATA%" ^
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

%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "checkpoints\%CHECKPOINT%" ^
    --output-dir "%INFERENCE_DIR%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Inference examples failed for %MODEL% at %EPOCHS% epochs.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
