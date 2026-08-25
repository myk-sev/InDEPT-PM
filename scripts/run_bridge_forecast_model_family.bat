@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Usage: scripts\run_bridge_forecast_model_family.bat [device] [batch_size] [workers]
rem Add --dry-run first to print the 44-run pretrained/control matrix.
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
)

set "DEVICE=%~1"
set "BATCH_SIZE=%~2"
set "WORKERS=%~3"
if not defined DEVICE set "DEVICE=auto"
if not defined BATCH_SIZE set "BATCH_SIZE=64"
if not defined WORKERS set "WORKERS=0"

set "START_DIR=%CD%"
cd /d "%REPO_ROOT%" || exit /b 1

set "PYTHON=.venv\Scripts\python.exe"
set "TRAINER=pm25_transformer.py"
set "EVALUATOR=evaluate_bridge_forecast.py"
set "INFERENCE_ROOT=inference"
set "CHECKPOINT_ROOT=%INFERENCE_ROOT%\checkpoints"
set "METRICS_ROOT=%INFERENCE_ROOT%\metrics"
set "GRAPH_ROOT=%INFERENCE_ROOT%\graphs"
set "FORECAST_ROOT=%INFERENCE_ROOT%\forecasts"
set "EVALUATION_ROOT=%INFERENCE_ROOT%\evaluations"
set "CACHE=inference\caches\keller_elementary_school_cache_cyclical.pt"
if not exist "%CACHE%" set "CACHE=inference\keller_elementary_school_cache_cyclical.pt"
set "ALL_DATA=inputs\unmasked_type\all_old_training_data_exclusion_informed_finetuned_cyclical.csv"
set "K12_DATA=inputs\unmasked_type\school_old_training_data_exclusion_informed_finetuned_cyclical.csv"

if not defined DRY_RUN (
    for %%P in ("%PYTHON%" "%TRAINER%" "%EVALUATOR%" "%CACHE%" "%ALL_DATA%" "%K12_DATA%") do (
        if not exist "%%~P" (
            echo Required path not found: %%~P
            goto :fail
        )
    )
)

set "TRAINING_COUNT=0"
call :run_dataset "all_excl_fine_t" "%ALL_DATA%" "all_excl_final" || goto :fail
call :run_dataset "k12_excl_fine_t" "%K12_DATA%" "k12_excl_final" || goto :fail

echo Completed %TRAINING_COUNT% bridge forecast training runs.
cd /d "%START_DIR%"
exit /b 0

:run_dataset
set "DATASET=%~1"
set "TRAINING_DATA=%~2"
set "LEGACY_DATASET=%~3"
echo Dataset: %DATASET%
echo Training data: %TRAINING_DATA%
for /f "delims=" %%M in ('%PYTHON% -c "from masked_pretraining.models import model_names; print(*model_names(), sep='\n')"') do (
    call :run_pair "%%M" || exit /b 1
)
exit /b 0

:run_pair
set "SOURCE_MODEL=%~1"
set "BRIDGE_CHECKPOINT=%CHECKPOINT_ROOT%\%DATASET%_%SOURCE_MODEL%.pt"
if not exist "%BRIDGE_CHECKPOINT%" set "BRIDGE_CHECKPOINT=%CHECKPOINT_ROOT%\bridge_training__%LEGACY_DATASET%__%SOURCE_MODEL%.pt"
if not defined DRY_RUN if not exist "%BRIDGE_CHECKPOINT%" (
    echo Bridge checkpoint not found: %BRIDGE_CHECKPOINT%
    exit /b 1
)
call :run_one "pretrained" 3 || exit /b 1
call :run_one "random-control" 0 || exit /b 1
exit /b 0

:run_one
set "INITIALIZATION=%~1"
set "FREEZE_EPOCHS=%~2"
set "MODEL=bridge-forecast-%SOURCE_MODEL%"
set "ARTIFACT_NAME=%DATASET%_%MODEL%-%INITIALIZATION%"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "FORECAST_DIR=%FORECAST_ROOT%\%ARTIFACT_NAME%"
set "EVALUATION=%EVALUATION_ROOT%\%ARTIFACT_NAME%.json"
set /a TRAINING_COUNT+=1 >nul
echo.
echo Starting %DATASET% / %SOURCE_MODEL% / %INITIALIZATION%
echo Bridge checkpoint: %BRIDGE_CHECKPOINT%
echo Forecast checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Forecasts: %FORECAST_DIR%
echo Evaluation: %EVALUATION%
if defined DRY_RUN exit /b 0

set "INITIALIZATION_VALUE=%INITIALIZATION%"
if /i "%INITIALIZATION%"=="random-control" set "INITIALIZATION_VALUE=random"
%PYTHON% %TRAINER% train ^
    --model "%MODEL%" ^
    --pretrained-checkpoint "%BRIDGE_CHECKPOINT%" ^
    --history-initialization "%INITIALIZATION_VALUE%" ^
    --training-data "%TRAINING_DATA%" ^
    --epochs 50 ^
    --freeze-history-epochs %FREEZE_EPOCHS% ^
    --forecast-horizons 3 6 12 24 36 ^
    --horizon-stage-epochs 5 5 10 10 20 ^
    --early-stopping-patience 20 ^
    --batch-size %BATCH_SIZE% ^
    --num-workers %WORKERS% ^
    --device "%DEVICE%" ^
    --metrics-output "%METRICS%" ^
    --loss-plot "%LOSS_CURVE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Forecast training failed for %DATASET% / %SOURCE_MODEL% / %INITIALIZATION%.
    exit /b 1
)

%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --output-dir "%FORECAST_DIR%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Forecast inference failed for %DATASET% / %SOURCE_MODEL% / %INITIALIZATION%.
    exit /b 1
)

%PYTHON% %EVALUATOR% ^
    --checkpoint "%CHECKPOINT%" ^
    --output "%EVALUATION%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Forecast evaluation failed for %DATASET% / %SOURCE_MODEL% / %INITIALIZATION%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
