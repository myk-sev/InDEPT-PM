@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Trains the four core forecast architectures from random initialization.
rem No reconstruction or bridge checkpoint is opened.
rem Usage: scripts\run_four_core_24h_forecast_from_scratch.bat [--dry-run] [device] [batch_size] [workers]
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
set "TRAINING_DATA=inputs\forecasting\all_old_training_data_exclusion_informed_finetuned_cyclical.csv"
set "CACHE=inference\caches\keller_elementary_school_cache_cyclical.pt"
if not exist "%CACHE%" set "CACHE=inference\keller_elementary_school_cache_cyclical.pt"
set "CHECKPOINT_ROOT=inference\checkpoints"
set "METRICS_ROOT=inference\metrics"
set "GRAPH_ROOT=inference\graphs"
set "REPORT_ROOT=inference\reports"
set "FORECAST_ROOT=inference\forecasts"

for %%P in ("%PYTHON%" "%TRAINER%" "%TRAINING_DATA%" "%CACHE%") do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

set "TRAINING_COUNT=0"
for %%M in (
    gru
    single-self-attention-encoder
    dual-encoder-cross-fusion
    dual-encoder-cross-fusion-outdoor-availability-recency
) do (
    call :run_model "%%M" || goto :fail
)

echo Completed %TRAINING_COUNT% fresh core 24-hour forecast training runs.
cd /d "%START_DIR%"
exit /b 0

:run_model
set "SOURCE_MODEL=%~1"
set "FORECAST_MODEL=bridge-forecast-%SOURCE_MODEL%"
set "ARTIFACT_NAME=all_excl_fine_t_%FORECAST_MODEL%-from-scratch-24h"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "RECOVERY_CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.last.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "REPORT=%REPORT_ROOT%\%ARTIFACT_NAME%.csv"
set "FORECAST_DIR=%FORECAST_ROOT%\%ARTIFACT_NAME%"
set /a TRAINING_COUNT+=1 >nul

echo.
echo Run %TRAINING_COUNT% of 4
echo History model: %SOURCE_MODEL%
echo Forecast model: %FORECAST_MODEL%
echo History initialization: random
echo Prediction window: 24 hours
echo Epochs: 50
echo Training data: %TRAINING_DATA%
echo Forecast checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Final report: %REPORT%
echo Inference examples: %FORECAST_DIR%
if defined DRY_RUN exit /b 0

set "RESUME_FLAG="
if exist "%CHECKPOINT%" set "RESUME_FLAG=--resume"
if exist "%RECOVERY_CHECKPOINT%" set "RESUME_FLAG=--resume"

%PYTHON% %TRAINER% train ^
    --model "%FORECAST_MODEL%" ^
    --history-initialization random ^
    --training-data "%TRAINING_DATA%" ^
    --prediction-hours 24 ^
    --epochs 50 ^
    --forecast-horizons 3 6 12 24 ^
    --horizon-stage-epochs 5 5 10 30 ^
    --early-stopping-patience 0 ^
    --batch-size "%BATCH_SIZE%" ^
    --num-workers "%WORKERS%" ^
    --device "%DEVICE%" ^
    --metrics-output "%METRICS%" ^
    --loss-plot "%LOSS_CURVE%" ^
    --report-output "%REPORT%" ^
    --checkpoint "%CHECKPOINT%" %RESUME_FLAG%
if errorlevel 1 (
    echo Fresh forecast training failed for %SOURCE_MODEL%.
    exit /b 1
)

%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --output-dir "%FORECAST_DIR%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Fresh forecast inference failed for %SOURCE_MODEL%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
