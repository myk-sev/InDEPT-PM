@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Usage: scripts\run_base_reconstruction_model_family.bat [device] [epochs_per_stage] [batch_size] [workers] [torch_threads]
rem Add --dry-run first to print the 22-run matrix without starting training.
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
set "INFERENCE_ROOT=inference"
set "CHECKPOINT_ROOT=%INFERENCE_ROOT%\checkpoints"
set "METRICS_ROOT=%INFERENCE_ROOT%\metrics"
set "GRAPH_ROOT=%INFERENCE_ROOT%\graphs"
set "RECONSTRUCTION_ROOT=%INFERENCE_ROOT%\reconstructions"
set "ALL_DATA=inputs\all_sensors_exclusion_informed_finetuned_masked_training_data.csv"
set "K12_DATA=inputs\k12_exclusion_informed_finetuned_masked_training_data.csv"

if not defined DRY_RUN (
    for %%P in ("%PYTHON%" "%ALL_DATA%" "%K12_DATA%") do (
        if not exist "%%~P" (
            echo Required path not found: %%~P
            goto :fail
        )
    )
)

set "TRAINING_COUNT=0"
call :run_dataset "all_excl_fine_t" "%ALL_DATA%" || goto :fail
call :run_dataset "k12_excl_fine_t" "%K12_DATA%" || goto :fail

echo Completed %TRAINING_COUNT% base-reconstruction training runs.
cd /d "%START_DIR%"
exit /b 0

:run_dataset
set "DATASET=%~1"
set "TRAINING_DATA=%~2"
echo Dataset: %DATASET%
echo Training data: %TRAINING_DATA%
if not defined DRY_RUN (
    %PYTHON% -m masked_pretraining audit ^
        --training-data "%TRAINING_DATA%" >nul
    if errorlevel 1 exit /b 1
)

for /f "delims=" %%M in ('%PYTHON% -c "from masked_pretraining.models import model_names; print(*model_names(), sep='\n')"') do (
    call :run_model "%%M" || exit /b 1
)
exit /b 0

:run_model
set "MODEL=%~1"
set "ARTIFACT_NAME=%DATASET%_%MODEL%"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "RECONSTRUCTION_DIR=%RECONSTRUCTION_ROOT%\%ARTIFACT_NAME%"
set "RECONSTRUCTION_OUTPUT=%RECONSTRUCTION_DIR%\run.reconstruction_examples.png"
set /a TRAINING_COUNT+=1 >nul
echo.
echo Starting %DATASET% / %MODEL%
echo Checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Reconstructions: %RECONSTRUCTION_DIR%
if defined DRY_RUN exit /b 0

%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    --stages points short_blocks mixed_blocks cross_channel suffix_3 suffix_6 suffix_12 ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-every-epochs 5 ^
    --reconstruction-output "%RECONSTRUCTION_OUTPUT%" ^
    --loss-curve-output "%LOSS_CURVE%" ^
    --metrics-output "%METRICS%" ^
    --final-checkpoint-only ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --torch-threads "%TORCH_THREADS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Training failed for %DATASET% / %MODEL%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
