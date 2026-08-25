@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Usage: scripts\run_exclusion_informed_finetuned_non_masked_training.bat [device] [batch_size] [num_workers]
rem Add --dry-run first to print the planned runs without starting training.
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
set "LINEAR_CACHE=inference\caches\keller_elementary_school_cache.pt"
set "CYCLICAL_CACHE=inference\caches\keller_elementary_school_cache_cyclical.pt"
if not exist "%LINEAR_CACHE%" set "LINEAR_CACHE=inference\keller_elementary_school_cache.pt"
if not exist "%CYCLICAL_CACHE%" set "CYCLICAL_CACHE=inference\keller_elementary_school_cache_cyclical.pt"
set "INFERENCE_ROOT=inference"
set "CHECKPOINT_ROOT=%INFERENCE_ROOT%\checkpoints"
set "METRICS_ROOT=%INFERENCE_ROOT%\metrics"
set "GRAPH_ROOT=%INFERENCE_ROOT%\graphs"
set "FORECAST_ROOT=%INFERENCE_ROOT%\forecasts"
set "SCHOOL_DATA=inputs\unmasked_type\school_old_training_data_exclusion_informed_finetuned.csv"
set "SCHOOL_CYCLICAL_DATA=inputs\unmasked_type\k12_exclusion_informed_finetuned_tempo_naqfc_forecast_training_cyclical.csv"
set "ALL_DATA=inputs\unmasked_type\all_old_training_data_exclusion_informed_finetuned.csv"
set "ALL_CYCLICAL_DATA=inputs\unmasked_type\all_old_training_data_exclusion_informed_finetuned_cyclical.csv"

for %%P in (
    "%PYTHON%"
    "%TRAINER%"
    "%LINEAR_CACHE%"
    "%CYCLICAL_CACHE%"
    "%SCHOOL_DATA%"
    "%SCHOOL_CYCLICAL_DATA%"
    "%ALL_DATA%"
    "%ALL_CYCLICAL_DATA%"
) do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

set "TRAINING_COUNT=0"
call :run_dataset "school" "%SCHOOL_DATA%" "%SCHOOL_CYCLICAL_DATA%" || goto :fail
call :run_dataset "all_sensors" "%ALL_DATA%" "%ALL_CYCLICAL_DATA%" || goto :fail

echo Completed %TRAINING_COUNT% training runs.
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
set "TRAINING_DATA=%LINEAR_DATA%"
set "CACHE=%LINEAR_CACHE%"
if not "%MODEL:cyclical=%"=="%MODEL%" (
    set "TRAINING_DATA=%CYCLICAL_DATA%"
    set "CACHE=%CYCLICAL_CACHE%"
)
set "ARTIFACT_NAME=%DATASET%_excl_fine_t_%MODEL%_%EPOCHS%ep"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "FORECAST_DIR=%FORECAST_ROOT%\%ARTIFACT_NAME%"
set /a TRAINING_COUNT+=1 >nul
echo Starting %DATASET% / %MODEL%: epochs=%EPOCHS% patience=%EPOCHS% data=%TRAINING_DATA% checkpoint=%CHECKPOINT%
echo Checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Forecasts: %FORECAST_DIR%
if defined DRY_RUN exit /b 0

%PYTHON% %TRAINER% train ^
    --model "%MODEL%" ^
    --training-data "%TRAINING_DATA%" ^
    --epochs %EPOCHS% ^
    --early-stopping-patience %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --num-workers %NUM_WORKERS% ^
    --device "%DEVICE%" ^
    --metrics-output "%METRICS%" ^
    --loss-plot "%LOSS_CURVE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Training failed for %DATASET% / %MODEL% at %EPOCHS% epochs.
    exit /b 1
)

%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --output-dir "%FORECAST_DIR%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Inference examples failed for %DATASET% / %MODEL% at %EPOCHS% epochs.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
