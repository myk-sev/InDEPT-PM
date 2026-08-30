@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Transfers every completed hyperparameter bridge history into its matching
rem forecaster, then runs supervised training, inference, and evaluation.
rem Usage: scripts\run_all_sensor_reconstruction_hyperparameter_grid_forecast.bat [--dry-run] [device] [batch_size] [workers]
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
set "TRAINING_DATA=inputs\forecasting\all_old_training_data_exclusion_informed_finetuned_cyclical.csv"
set "CACHE=inference\caches\keller_elementary_school_cache_cyclical.pt"
if not exist "%CACHE%" set "CACHE=inference\keller_elementary_school_cache_cyclical.pt"
set "CHECKPOINT_ROOT=inference\checkpoints"
set "METRICS_ROOT=inference\metrics"
set "GRAPH_ROOT=inference\graphs"
set "REPORT_ROOT=inference\reports"
set "FORECAST_ROOT=inference\forecasts"
set "EVALUATION_ROOT=inference\evaluations"

set "CHECKPOINT_COUNT=0"
if not defined DRY_RUN (
    for %%P in ("%PYTHON%" "%TRAINER%" "%EVALUATOR%" "%TRAINING_DATA%" "%CACHE%") do (
        if not exist "%%~P" (
            echo Required path not found: %%~P
            goto :fail
        )
    )
    set "MODE=preflight"
    call :for_each_configuration || goto :fail
    echo Verified %CHECKPOINT_COUNT% completed hyperparameter bridge checkpoints.
)

set "MODE=train"
set "TRAINING_COUNT=0"
call :for_each_configuration || goto :fail

echo Completed %TRAINING_COUNT% hyperparameter forecast training runs.
cd /d "%START_DIR%"
exit /b 0

:for_each_configuration
for %%R in (2e-4 1e-4) do (
    call :process_configuration "learning-rate" "%%R" "%%R" "64" "3" "4" || exit /b 1
)
for %%D in (16 32 64 128 256) do (
    call :process_configuration "model-dim" "%%D" "3e-4" "%%D" "3" "4" || exit /b 1
)
for %%L in (1 2 3 4 5 6) do (
    call :process_configuration "transformer-depth" "%%L" "3e-4" "64" "%%L" "4" || exit /b 1
)
for %%H in (1 2 4 8 16) do (
    call :process_configuration "head-size" "%%H" "3e-4" "64" "3" "%%H" || exit /b 1
)
exit /b 0

:process_configuration
set "SWEEP=%~1"
set "VALUE=%~2"
set "LEARNING_RATE=%~3"
set "MODEL_DIM=%~4"
set "LAYERS=%~5"
set "HEADS=%~6"

for %%M in (
    gru
    single-self-attention-encoder
    dual-encoder-cross-fusion
    dual-encoder-cross-fusion-outdoor-availability-recency
) do (
    call :set_model "%%M"
    if /i "%MODE%"=="preflight" (
        call :check_checkpoint || exit /b 1
    ) else (
        call :run_model || exit /b 1
    )
)
exit /b 0

:set_model
set "SOURCE_MODEL=%~1"
set "SOURCE_ARTIFACT=all_excl_fine_t_hp_%SWEEP%_%VALUE%_lr%LEARNING_RATE%_dim%MODEL_DIM%_depth%LAYERS%_heads%HEADS%_%SOURCE_MODEL%"
set "BRIDGE_CHECKPOINT=%CHECKPOINT_ROOT%\%SOURCE_ARTIFACT%.pt"
set "FORECAST_MODEL=bridge-forecast-%SOURCE_MODEL%"
set "ARTIFACT_NAME=all_excl_fine_t_hp_%SWEEP%_%VALUE%_lr%LEARNING_RATE%_dim%MODEL_DIM%_depth%LAYERS%_heads%HEADS%_%FORECAST_MODEL%-pretrained"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "RECOVERY_CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.last.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "REPORT=%REPORT_ROOT%\%ARTIFACT_NAME%.csv"
set "FORECAST_DIR=%FORECAST_ROOT%\%ARTIFACT_NAME%"
set "EVALUATION=%EVALUATION_ROOT%\%ARTIFACT_NAME%.json"
exit /b 0

:check_checkpoint
if not exist "%BRIDGE_CHECKPOINT%" (
    echo Bridge checkpoint not found: %BRIDGE_CHECKPOINT%
    exit /b 1
)
%PYTHON% -c "import math, sys; from pathlib import Path; from pm25_models import load_bridge_checkpoint; checkpoint=load_bridge_checkpoint(Path(sys.argv[1])); metadata=checkpoint['metadata']; config=metadata['model_config']; assert metadata['model_name'] == sys.argv[2], 'model mismatch'; assert config['model_dim'] == int(sys.argv[3]), 'model-dimension mismatch'; assert config['layers'] == int(sys.argv[4]), 'depth mismatch'; assert config['heads'] == int(sys.argv[5]), 'head-count mismatch'; assert math.isclose(checkpoint['optimizer_state']['param_groups'][0]['lr'], float(sys.argv[6])), 'learning-rate mismatch'" "%BRIDGE_CHECKPOINT%" "%SOURCE_MODEL%" "%MODEL_DIM%" "%LAYERS%" "%HEADS%" "%LEARNING_RATE%"
if errorlevel 1 exit /b 1
set /a CHECKPOINT_COUNT+=1 >nul
exit /b 0

:run_model
set /a TRAINING_COUNT+=1 >nul
echo.
echo Run %TRAINING_COUNT% of 72
echo Sweep: %SWEEP%=%VALUE%
echo History model: %SOURCE_MODEL%
echo Bridge checkpoint: %BRIDGE_CHECKPOINT%
echo Forecast checkpoint: %CHECKPOINT%
echo Evaluation: %EVALUATION%
if defined DRY_RUN exit /b 0

set "RESUME_FLAG="
if exist "%CHECKPOINT%" set "RESUME_FLAG=--resume"
if exist "%RECOVERY_CHECKPOINT%" set "RESUME_FLAG=--resume"

%PYTHON% %TRAINER% train ^
    --model "%FORECAST_MODEL%" ^
    --pretrained-checkpoint "%BRIDGE_CHECKPOINT%" ^
    --history-initialization pretrained ^
    --training-data "%TRAINING_DATA%" ^
    --epochs 50 ^
    --freeze-history-epochs 3 ^
    --forecast-horizons 3 6 12 24 36 ^
    --horizon-stage-epochs 5 5 10 10 20 ^
    --early-stopping-patience 20 ^
    --batch-size "%BATCH_SIZE%" ^
    --num-workers "%WORKERS%" ^
    --device "%DEVICE%" ^
    --metrics-output "%METRICS%" ^
    --loss-plot "%LOSS_CURVE%" ^
    --report-output "%REPORT%" ^
    --checkpoint "%CHECKPOINT%" %RESUME_FLAG%
if errorlevel 1 (
    echo Forecast training failed for %SOURCE_ARTIFACT%.
    exit /b 1
)

%PYTHON% -m inference.run_cached_inference ^
    --cache "%CACHE%" ^
    --checkpoint "%CHECKPOINT%" ^
    --output-dir "%FORECAST_DIR%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Forecast inference failed for %SOURCE_ARTIFACT%.
    exit /b 1
)

%PYTHON% %EVALUATOR% ^
    --checkpoint "%CHECKPOINT%" ^
    --output "%EVALUATION%" ^
    --device "%DEVICE%"
if errorlevel 1 (
    echo Forecast evaluation failed for %SOURCE_ARTIFACT%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
