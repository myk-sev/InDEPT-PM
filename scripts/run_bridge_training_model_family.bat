@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Usage: scripts\run_bridge_training_model_family.bat [device] [epochs_per_stage] [batch_size] [workers] [torch_threads]
rem Add --dry-run first to print the 22-run bridge matrix without starting training.
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
set "LEGACY_BASE_ROOT=masked_pretraining\runs\base_reconstruction"
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

if not defined DRY_RUN (
    set "CHECKPOINT_COUNT=0"
    call :preflight_dataset "all_excl_fine_t" "%ALL_DATA%" "all_excl_final" || goto :fail
    call :preflight_dataset "k12_excl_fine_t" "%K12_DATA%" "k12_excl_final" || goto :fail
    echo Verified %CHECKPOINT_COUNT% complete base checkpoints.
)

set "TRAINING_COUNT=0"
call :run_dataset "all_excl_fine_t" "%ALL_DATA%" "all_excl_final" || goto :fail
call :run_dataset "k12_excl_fine_t" "%K12_DATA%" "k12_excl_final" || goto :fail

if not defined DRY_RUN (
    %PYTHON% -m masked_pretraining.verify_bridge_family ^
        --checkpoint-root "%CHECKPOINT_ROOT%" || goto :fail
)

echo Completed %TRAINING_COUNT% bridge training runs.
cd /d "%START_DIR%"
exit /b 0

:preflight_dataset
set "DATASET=%~1"
set "TRAINING_DATA=%~2"
set "LEGACY_DATASET=%~3"
for /f "delims=" %%M in ('%PYTHON% -c "from masked_pretraining.models import model_names; print(*model_names(), sep='\n')"') do (
    call :check_base_checkpoint "%%M" || exit /b 1
)
exit /b 0

:check_base_checkpoint
set "MODEL=%~1"
set "BASE_CHECKPOINT=%CHECKPOINT_ROOT%\%DATASET%_%MODEL%.pt"
if not exist "%BASE_CHECKPOINT%" set "BASE_CHECKPOINT=%CHECKPOINT_ROOT%\base_reconstruction__%LEGACY_DATASET%__%MODEL%.pt"
if not exist "%BASE_CHECKPOINT%" set "BASE_CHECKPOINT=%LEGACY_BASE_ROOT%\%LEGACY_DATASET%\%MODEL%\checkpoints\run.pt"
if not exist "%BASE_CHECKPOINT%" (
    echo Base checkpoint not found: %BASE_CHECKPOINT%
    exit /b 1
)
%PYTHON% -c "import sys, torch; from pathlib import Path; from masked_pretraining.data import file_sha256; from masked_pretraining.masking import STAGES; checkpoint=torch.load(sys.argv[1], map_location='cpu', weights_only=False); metadata=checkpoint.get('metadata', {}); missing=[stage for stage in STAGES if stage not in metadata.get('completed_stages', ())]; assert not missing, 'missing base stages: ' + ', '.join(missing); assert metadata.get('model_name') == sys.argv[2], 'model mismatch'; assert metadata.get('training_data_sha256') == file_sha256(Path(sys.argv[3])), 'training-data hash mismatch'; assert 'optimizer_state' in checkpoint, 'optimizer state missing'" "%BASE_CHECKPOINT%" "%MODEL%" "%TRAINING_DATA%"
if errorlevel 1 exit /b 1
set /a CHECKPOINT_COUNT+=1 >nul
exit /b 0

:run_dataset
set "DATASET=%~1"
set "TRAINING_DATA=%~2"
set "LEGACY_DATASET=%~3"
echo Dataset: %DATASET%
echo Training data: %TRAINING_DATA%
for /f "delims=" %%M in ('%PYTHON% -c "from masked_pretraining.models import model_names; print(*model_names(), sep='\n')"') do (
    call :run_model "%%M" || exit /b 1
)
exit /b 0

:run_model
set "MODEL=%~1"
set "BASE_CHECKPOINT=%CHECKPOINT_ROOT%\%DATASET%_%MODEL%.pt"
if not exist "%BASE_CHECKPOINT%" set "BASE_CHECKPOINT=%CHECKPOINT_ROOT%\base_reconstruction__%LEGACY_DATASET%__%MODEL%.pt"
if not exist "%BASE_CHECKPOINT%" set "BASE_CHECKPOINT=%LEGACY_BASE_ROOT%\%LEGACY_DATASET%\%MODEL%\checkpoints\run.pt"
set "ARTIFACT_NAME=%DATASET%_%MODEL%"
set "CHECKPOINT=%CHECKPOINT_ROOT%\%ARTIFACT_NAME%.pt"
set "METRICS=%METRICS_ROOT%\%ARTIFACT_NAME%.csv"
set "LOSS_CURVE=%GRAPH_ROOT%\%ARTIFACT_NAME%.png"
set "RECONSTRUCTION_DIR=%RECONSTRUCTION_ROOT%\%ARTIFACT_NAME%"
set "RECONSTRUCTION_OUTPUT=%RECONSTRUCTION_DIR%\run.reconstruction_examples.png"
set /a TRAINING_COUNT+=1 >nul
echo.
echo Starting bridge: %DATASET% / %MODEL%
echo Base checkpoint: %BASE_CHECKPOINT%
echo Checkpoint: %CHECKPOINT%
echo Metrics: %METRICS%
echo Loss graph: %LOSS_CURVE%
echo Reconstructions: %RECONSTRUCTION_DIR%
if defined DRY_RUN exit /b 0

%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    --resume "%BASE_CHECKPOINT%" ^
    --tempo-missingness-bridge ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-every-epochs 5 ^
    --reconstruction-output "%RECONSTRUCTION_OUTPUT%" ^
    --loss-curve-output "%LOSS_CURVE%" ^
    --metrics-output "%METRICS%" ^
    --resume-metrics ^
    --final-checkpoint-only ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --torch-threads "%TORCH_THREADS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%CHECKPOINT%"
if errorlevel 1 (
    echo Bridge training failed for %DATASET% / %MODEL%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
