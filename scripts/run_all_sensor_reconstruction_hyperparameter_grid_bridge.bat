@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Adds the three synthetic missingness bridge stages to every checkpoint
rem produced by run_all_sensor_reconstruction_hyperparameter_grid.bat.
rem Usage: scripts\run_all_sensor_reconstruction_hyperparameter_grid_bridge.bat [--dry-run] [device] [epochs_per_stage] [batch_size] [workers] [torch_threads]
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
)

set "DEVICE=%~1"
set "EPOCHS_PER_STAGE=%~2"
set "BATCH_SIZE=%~3"
set "WORKERS=%~4"
set "TORCH_THREADS=%~5"
if not defined DEVICE set "DEVICE=auto"
if not defined EPOCHS_PER_STAGE set "EPOCHS_PER_STAGE=20"
if not defined BATCH_SIZE set "BATCH_SIZE=64"
if not defined WORKERS set "WORKERS=0"
if not defined TORCH_THREADS set "TORCH_THREADS=0"

set "START_DIR=%CD%"
cd /d "%REPO_ROOT%" || exit /b 1

set "PYTHON=.venv\Scripts\python.exe"
set "TRAINING_DATA=inputs\reconstruction\all_sensors_exclusion_informed_finetuned_masked_training_data.csv"
set "CHECKPOINT_ROOT=inference\checkpoints"
set "METRICS_ROOT=inference\metrics"
set "GRAPH_ROOT=inference\graphs"
set "REPORT_ROOT=inference\reports"
set "RECONSTRUCTION_ROOT=inference\reconstructions"

set "CHECKPOINT_COUNT=0"
if not defined DRY_RUN (
    for %%P in ("%PYTHON%" "%TRAINING_DATA%") do (
        if not exist "%%~P" (
            echo Required path not found: %%~P
            goto :fail
        )
    )
    set "MODE=preflight"
    call :for_each_configuration || goto :fail
)
if not defined DRY_RUN echo Verified %CHECKPOINT_COUNT% complete hyperparameter checkpoints.

set "MODE=train"
set "TRAINING_COUNT=0"
call :for_each_configuration || goto :fail

echo Completed %TRAINING_COUNT% hyperparameter bridge training runs.
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
set "MODEL=%~1"
set "ARTIFACT_NAME=all_excl_fine_t_hp_%SWEEP%_%VALUE%_lr%LEARNING_RATE%_dim%MODEL_DIM%_depth%LAYERS%_heads%HEADS%_%MODEL%"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "REPORT=%REPORT_ROOT%\%ARTIFACT_NAME%.csv"
set "RECONSTRUCTION_DIR=%RECONSTRUCTION_ROOT%\%ARTIFACT_NAME%"
set "RECONSTRUCTION_OUTPUT=%RECONSTRUCTION_DIR%\run.reconstruction_examples.png"
exit /b 0

:check_checkpoint
if not exist "%CHECKPOINT%" (
    echo Base checkpoint not found: %CHECKPOINT%
    exit /b 1
)
%PYTHON% -c "import math, sys, torch; from pathlib import Path; from masked_pretraining.data import file_sha256; from masked_pretraining.masking import STAGES; checkpoint=torch.load(sys.argv[1], map_location='cpu', weights_only=False); metadata=checkpoint.get('metadata', {}); config=metadata.get('model_config', {}); missing=[stage for stage in STAGES if stage not in metadata.get('completed_stages', ())]; assert not missing, 'missing base stages: ' + ', '.join(missing); assert metadata.get('model_name') == sys.argv[2], 'model mismatch'; assert metadata.get('training_data_sha256') == file_sha256(Path(sys.argv[3])), 'training-data hash mismatch'; assert config.get('model_dim') == int(sys.argv[4]), 'model-dimension mismatch'; assert config.get('layers') == int(sys.argv[5]), 'depth mismatch'; assert config.get('heads') == int(sys.argv[6]), 'head-count mismatch'; assert 'optimizer_state' in checkpoint, 'optimizer state missing'; assert math.isclose(checkpoint['optimizer_state']['param_groups'][0]['lr'], float(sys.argv[7])), 'learning-rate mismatch'" "%CHECKPOINT%" "%MODEL%" "%TRAINING_DATA%" "%MODEL_DIM%" "%LAYERS%" "%HEADS%" "%LEARNING_RATE%"
if errorlevel 1 exit /b 1
set /a CHECKPOINT_COUNT+=1 >nul
exit /b 0

:run_model
set /a TRAINING_COUNT+=1 >nul
echo.
echo Run %TRAINING_COUNT% of 72
echo Sweep: %SWEEP%=%VALUE%
echo Learning rate: %LEARNING_RATE%  Model dimension: %MODEL_DIM%  Depth: %LAYERS%  Heads: %HEADS%
echo Model: %MODEL%
echo Resume checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Report: %REPORT%
echo Reconstructions: %RECONSTRUCTION_DIR%
if defined DRY_RUN exit /b 0

%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    --learning-rate "%LEARNING_RATE%" ^
    --model-dim "%MODEL_DIM%" ^
    --layers "%LAYERS%" ^
    --heads "%HEADS%" ^
    --resume "%CHECKPOINT%" ^
    --tempo-missingness-bridge ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-every-epochs 5 ^
    --reconstruction-output "%RECONSTRUCTION_OUTPUT%" ^
    --loss-curve-output "%LOSS_CURVE%" ^
    --metrics-output "%METRICS%" ^
    --report-output "%REPORT%" ^
    --resume-metrics ^
    --final-checkpoint-only ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --torch-threads "%TORCH_THREADS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Bridge training failed for %ARTIFACT_NAME%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
