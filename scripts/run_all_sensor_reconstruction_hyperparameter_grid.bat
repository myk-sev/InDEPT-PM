@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Varies one hyperparameter at a time across all four models before
rem advancing to the next value. Other parameters remain at their defaults:
rem learning rate 3e-4, model dimension 64, depth 3, and heads 4.
rem Usage: scripts\run_all_sensor_reconstruction_hyperparameter_grid.bat [--dry-run] [device] [epochs_per_stage] [batch_size] [workers] [torch_threads]
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

if not defined DRY_RUN (
    for %%P in ("%PYTHON%" "%TRAINING_DATA%") do (
        if not exist "%%~P" (
            echo Required path not found: %%~P
            goto :fail
        )
    )
    %PYTHON% -m masked_pretraining audit --training-data "%TRAINING_DATA%" >nul || goto :fail
)

set "TRAINING_COUNT=0"
for %%R in (2e-4 1e-4) do (
    call :run_configuration "learning-rate" "%%R" "%%R" "64" "3" "4" || goto :fail
)
for %%D in (16 32 64 128 256) do (
    call :run_configuration "model-dim" "%%D" "3e-4" "%%D" "3" "4" || goto :fail
)
for %%L in (1 2 3 4 5 6) do (
    call :run_configuration "transformer-depth" "%%L" "3e-4" "64" "%%L" "4" || goto :fail
)
for %%H in (1 2 4 8 16) do (
    call :run_configuration "head-size" "%%H" "3e-4" "64" "3" "%%H" || goto :fail
)

echo Completed %TRAINING_COUNT% hyperparameter training runs.
cd /d "%START_DIR%"
exit /b 0

:run_configuration
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
    call :run_model "%%M" || exit /b 1
)
exit /b 0

:run_model
set "MODEL=%~1"
set "ARTIFACT_NAME=all_excl_fine_t_hp_%SWEEP%_%VALUE%_lr%LEARNING_RATE%_dim%MODEL_DIM%_depth%LAYERS%_heads%HEADS%_%MODEL%"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "REPORT=%REPORT_ROOT%\%ARTIFACT_NAME%.csv"
set "RECONSTRUCTION_DIR=%RECONSTRUCTION_ROOT%\%ARTIFACT_NAME%"
set "RECONSTRUCTION_OUTPUT=%RECONSTRUCTION_DIR%\run.reconstruction_examples.png"
set /a TRAINING_COUNT+=1 >nul

echo.
echo Run %TRAINING_COUNT% of 72
echo Sweep: %SWEEP%=%VALUE%
echo Learning rate: %LEARNING_RATE%  Model dimension: %MODEL_DIM%  Depth: %LAYERS%  Heads: %HEADS%
echo Model: %MODEL%
echo Checkpoint: %CHECKPOINT%
if defined DRY_RUN exit /b 0

%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    --learning-rate "%LEARNING_RATE%" ^
    --model-dim "%MODEL_DIM%" ^
    --layers "%LAYERS%" ^
    --heads "%HEADS%" ^
    --stages points short_blocks mixed_blocks cross_channel suffix_3 suffix_6 suffix_12 ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-every-epochs 5 ^
    --reconstruction-output "%RECONSTRUCTION_OUTPUT%" ^
    --loss-curve-output "%LOSS_CURVE%" ^
    --metrics-output "%METRICS%" ^
    --report-output "%REPORT%" ^
    --final-checkpoint-only ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --torch-threads "%TORCH_THREADS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Training failed for %ARTIFACT_NAME%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
