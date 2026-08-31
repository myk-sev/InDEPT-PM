@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Trains the four core bridge-forecast models through an 18-hour horizon.
rem Uses the model-dim=64 sweep run: learning rate 3e-4, dimension 64, depth 3, heads 4.
rem Usage: scripts\run_four_core_18h_forecast_models.bat [--dry-run] [device] [batch_size] [workers]
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
set "HYPERPARAMETER_STEM=all_excl_fine_t_hp_model-dim_64_lr3e-4_dim64_depth3_heads4"

for %%P in ("%PYTHON%" "%TRAINER%" "%TRAINING_DATA%" "%CACHE%") do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

if not defined DRY_RUN (
    call :check_inference_contract || goto :fail
    set "MODE=preflight"
    call :for_each_model || goto :fail
)
set "MODE=train"
set "TRAINING_COUNT=0"
call :for_each_model || goto :fail

echo Completed %TRAINING_COUNT% core 18-hour forecast training runs.
cd /d "%START_DIR%"
exit /b 0

:for_each_model
for %%M in (
    gru
    single-self-attention-encoder
    dual-encoder-cross-fusion
    dual-encoder-cross-fusion-outdoor-availability-recency
) do (
    call :set_model "%%M"
    if /i "%MODE%"=="preflight" (
        call :check_inputs || exit /b 1
    ) else (
        call :run_model || exit /b 1
    )
)
exit /b 0

:set_model
set "SOURCE_MODEL=%~1"
set "SOURCE_ARTIFACT=%HYPERPARAMETER_STEM%_%SOURCE_MODEL%"
set "PRETRAINED_CHECKPOINT=%CHECKPOINT_ROOT%\%SOURCE_ARTIFACT%.pt"
set "HISTORY_METRICS=%METRICS_ROOT%\%SOURCE_ARTIFACT%.csv"
set "FORECAST_MODEL=bridge-forecast-%SOURCE_MODEL%"
set "ARTIFACT_NAME=%HYPERPARAMETER_STEM%_%FORECAST_MODEL%-pretrained-18h"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "RECOVERY_CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.last.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "REPORT=%REPORT_ROOT%\%ARTIFACT_NAME%.csv"
set "FORECAST_DIR=%FORECAST_ROOT%\%ARTIFACT_NAME%"
exit /b 0

:check_inputs
for %%P in ("%PRETRAINED_CHECKPOINT%" "%HISTORY_METRICS%") do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        exit /b 1
    )
)
exit /b 0

:run_model
set /a TRAINING_COUNT+=1 >nul
echo.
echo Run %TRAINING_COUNT% of 4
echo History model: %SOURCE_MODEL%
echo Forecast model: %FORECAST_MODEL%
echo Prediction window: 18 hours
echo Epochs: 50
echo Training data: %TRAINING_DATA%
echo Pretrained checkpoint: %PRETRAINED_CHECKPOINT%
echo Forecast checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Final report: %REPORT%
echo Inference examples: %FORECAST_DIR%
if defined DRY_RUN exit /b 0

set "RESUME_FLAG="
set "RESUME_CHECKPOINT="
if exist "%CHECKPOINT%" set "RESUME_CHECKPOINT=%CHECKPOINT%"
if exist "%RECOVERY_CHECKPOINT%" set "RESUME_CHECKPOINT=%RECOVERY_CHECKPOINT%"
if defined RESUME_CHECKPOINT (
    call :checkpoint_complete "%RESUME_CHECKPOINT%" 50
    if not errorlevel 1 (
        echo Training already completed; continuing with inference.
        goto :run_inference
    )
    set "RESUME_FLAG=--resume"
)

%PYTHON% %TRAINER% train ^
    --model "%FORECAST_MODEL%" ^
    --pretrained-checkpoint "%PRETRAINED_CHECKPOINT%" ^
    --history-initialization pretrained ^
    --history-metrics "%HISTORY_METRICS%" ^
    --training-data "%TRAINING_DATA%" ^
    --prediction-hours 18 ^
    --epochs 50 ^
    --freeze-history-epochs 3 ^
    --forecast-horizons 3 6 12 18 ^
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
    echo Forecast training failed for %SOURCE_MODEL%.
    exit /b 1
)

:run_inference
%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --output-dir "%FORECAST_DIR%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Forecast inference failed for %SOURCE_MODEL%.
    exit /b 1
)
exit /b 0

:check_inference_contract
%PYTHON% -c "from types import SimpleNamespace; import torch; from inference.run_cached_inference import validate_data_contract; config=SimpleNamespace(history_hours=168, prediction_hours=18, cyclical_time=True); sample={'history': torch.zeros(168, 8), 'forecast': torch.zeros(36, 7), 'target': torch.zeros(36)}; cache={'data_contract': {'history_shape': (168, 8), 'forecast_shape': (36, 7), 'target_shape': (36,), 'cyclical_time': True}}; validate_data_contract(cache, [{'sample_index': 0, 'sample': sample}], config)" >nul 2>nul
if errorlevel 1 (
    echo inference\run_cached_inference.py does not support an 18-hour prefix of the wider cache.
    echo Update inference\run_cached_inference.py before starting training.
    exit /b 1
)
exit /b 0

:checkpoint_complete
%PYTHON% -c "import sys, torch; checkpoint=torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(int(checkpoint.get('epoch', 0)) < int(sys.argv[2]))" "%~1" "%~2" >nul
exit /b %ERRORLEVEL%

:fail
cd /d "%START_DIR%"
exit /b 1
