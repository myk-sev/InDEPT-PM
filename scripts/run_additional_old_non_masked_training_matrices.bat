@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Usage: scripts\run_additional_old_non_masked_training_matrices.bat [device] [batch_size] [num_workers]
rem Add --dry-run first to print the 168 runs without starting training.
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
cd /d "%REPO_ROOT%" || exit /b 1

set "PYTHON=.venv\Scripts\python.exe"
set "TRAINER=pm25_transformer.py"
set "LINEAR_CACHE=inference\keller_elementary_school_cache.pt"
set "CYCLICAL_CACHE=inference\keller_elementary_school_cache_cyclical.pt"
set "INFERENCE_ROOT=inference\additional_old_non_masked_matrices"

for %%P in (
    "%PYTHON%"
    "%TRAINER%"
    "%LINEAR_CACHE%"
    "%CYCLICAL_CACHE%"
    "inputs\balanced_old_training_data.csv"
    "inputs\balanced_old_training_data_cyclical.csv"
    "inputs\non_school_old_training_data_exclusion_aware.csv"
    "inputs\non_school_old_training_data_exclusion_aware_cyclical.csv"
    "inputs\school_old_training_data.csv"
    "inputs\school_old_training_data_cyclical.csv"
    "inputs\school_old_training_data_exclusion_aware.csv"
    "inputs\school_old_training_data_exclusion_aware_cyclical.csv"
) do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

call :run_dataset "balanced" "inputs\balanced_old_training_data.csv" "inputs\balanced_old_training_data_cyclical.csv" || goto :fail
call :run_dataset "non-school-exclusion-aware" "inputs\non_school_old_training_data_exclusion_aware.csv" "inputs\non_school_old_training_data_exclusion_aware_cyclical.csv" || goto :fail
call :run_dataset "school" "inputs\school_old_training_data.csv" "inputs\school_old_training_data_cyclical.csv" || goto :fail
call :run_dataset "school-exclusion-aware" "inputs\school_old_training_data_exclusion_aware.csv" "inputs\school_old_training_data_exclusion_aware_cyclical.csv" || goto :fail

cd /d "%START_DIR%"
exit /b 0

:run_dataset
set "DATASET=%~1"
set "LINEAR_DATA=%~2"
set "CYCLICAL_DATA=%~3"
set "MODEL_COUNT=0"
echo Dataset: %DATASET%
for /f "delims=" %%M in ('%PYTHON% -c "from pm25_models import model_names; print(*model_names(), sep='\n')"') do (
    set /a MODEL_COUNT+=1 >nul
    call :run_model "%%M" || exit /b 1
)
if "%MODEL_COUNT%"=="0" (
    echo Could not read the available model names.
    exit /b 1
)
exit /b 0

:run_model
call :run_one "%~1" 5 || exit /b 1
call :run_one "%~1" 20 || exit /b 1
call :run_one "%~1" 100 || exit /b 1
exit /b 0

:run_one
set "MODEL=%~1"
set "EPOCHS=%~2"
set "CHECKPOINT=old-non-masked-%DATASET%-%MODEL%-%EPOCHS%ep.pt"
set "TRAINING_DATA=%LINEAR_DATA%"
set "CACHE=%LINEAR_CACHE%"
if not "%MODEL:cyclical=%"=="%MODEL%" (
    set "TRAINING_DATA=%CYCLICAL_DATA%"
    set "CACHE=%CYCLICAL_CACHE%"
)
set "INFERENCE_DIR=%INFERENCE_ROOT%\%DATASET%\%MODEL%\%EPOCHS%ep"
echo Starting %DATASET% / %MODEL%: epochs=%EPOCHS% patience=%EPOCHS% data=%TRAINING_DATA% checkpoint=%CHECKPOINT%
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
    echo Training failed for %DATASET% / %MODEL% at %EPOCHS% epochs.
    exit /b 1
)

%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "checkpoints\%CHECKPOINT%" ^
    --output-dir "%INFERENCE_DIR%" ^
    --loss-plot "graphs\%CHECKPOINT:.pt=.loss.png%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Inference examples failed for %DATASET% / %MODEL% at %EPOCHS% epochs.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
