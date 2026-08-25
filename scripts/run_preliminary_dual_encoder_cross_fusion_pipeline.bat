@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Runs the complete reconstruction and TEMPO-missingness curriculum for one model.
rem Uses only the K-12 exclusion-informed fine-tuned training contract.
rem Usage: scripts\run_preliminary_dual_encoder_cross_fusion_pipeline.bat [--dry-run] [device] [batch_size] [workers]
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
set "EPOCHS_PER_STAGE=30"

set "START_DIR=%CD%"
cd /d "%REPO_ROOT%" || exit /b 1

set "PYTHON=.venv\Scripts\python.exe"
set "MODEL=dual-encoder-cross-fusion"
set "CHECKPOINT_ROOT=inference\checkpoints"
set "GRAPH_ROOT=inference\graphs"
set "RECONSTRUCTION_ROOT=inference\reconstructions"
set "K12_DATA=inputs\reconstruction\k12_exclusion_informed_finetuned_masked_training_data.csv"

for %%P in ("%PYTHON%" "%K12_DATA%") do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

call :run_pipeline "k12_excl_final" "%K12_DATA%" || goto :fail
goto :success

:success
echo.
echo Preliminary %MODEL% pipeline complete for the K-12 cohort.
cd /d "%START_DIR%"
exit /b 0

:run_pipeline
set "DATASET=%~1"
set "TRAINING_DATA=%~2"
set "BASE_NAME=preliminary__base_reconstruction__%DATASET%__%MODEL%"
set "BRIDGE_NAME=preliminary__bridge_training__%DATASET%__%MODEL%"
set "BASE_CHECKPOINT=%CHECKPOINT_ROOT%\%BASE_NAME%.pt"
set "BRIDGE_CHECKPOINT=%CHECKPOINT_ROOT%\%BRIDGE_NAME%.pt"

echo.
echo Dataset: %DATASET%
echo Training data: %TRAINING_DATA%
if not defined DRY_RUN (
    %PYTHON% -m masked_pretraining audit --training-data "%TRAINING_DATA%" >nul || exit /b 1
)

for %%S in (points short_blocks mixed_blocks cross_channel suffix_3 suffix_6 suffix_12) do (
    call :run_base_stage "%%S" || exit /b 1
)
if not defined DRY_RUN call :require_base_curriculum || exit /b 1
for %%S in (tempo_bridge_50 tempo_bridge_70 tempo_bridge_86) do (
    call :run_bridge_stage "%%S" || exit /b 1
)
exit /b 0

:run_base_stage
set "STAGE=%~1"
call :stage_complete "%BASE_CHECKPOINT%" "%STAGE%"
if not errorlevel 1 (
    echo Skipping completed base stage: %DATASET% / %STAGE%
    exit /b 0
)
set "RESUME_ARGUMENT="
if exist "%BASE_CHECKPOINT%" set "RESUME_ARGUMENT=--resume "%BASE_CHECKPOINT%""
echo Starting base stage: %DATASET% / %STAGE%
if defined DRY_RUN exit /b 0
%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    %RESUME_ARGUMENT% ^
    --stages "%STAGE%" ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-output "%RECONSTRUCTION_ROOT%\%BASE_NAME%.%STAGE%.png" ^
    --loss-curve-output "%GRAPH_ROOT%\%BASE_NAME%.%STAGE%.loss_curve.png" ^
    --skip-metrics-csv ^
    --final-checkpoint-only ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%BASE_CHECKPOINT%"
exit /b %errorlevel%

:run_bridge_stage
set "STAGE=%~1"
call :stage_complete "%BRIDGE_CHECKPOINT%" "%STAGE%"
if not errorlevel 1 (
    echo Skipping completed bridge stage: %DATASET% / %STAGE%
    exit /b 0
)
set "RESUME_CHECKPOINT=%BASE_CHECKPOINT%"
if exist "%BRIDGE_CHECKPOINT%" set "RESUME_CHECKPOINT=%BRIDGE_CHECKPOINT%"
echo Starting bridge stage: %DATASET% / %STAGE%
if defined DRY_RUN exit /b 0
%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    --resume "%RESUME_CHECKPOINT%" ^
    --tempo-missingness-bridge ^
    --stages "%STAGE%" ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-output "%RECONSTRUCTION_ROOT%\%BRIDGE_NAME%.%STAGE%.png" ^
    --loss-curve-output "%GRAPH_ROOT%\%BRIDGE_NAME%.%STAGE%.loss_curve.png" ^
    --skip-metrics-csv ^
    --final-checkpoint-only ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%BRIDGE_CHECKPOINT%"
exit /b %errorlevel%

:stage_complete
if not exist "%~1" exit /b 1
%PYTHON% -c "import sys, torch; checkpoint=torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(0 if sys.argv[2] in checkpoint.get('metadata', {}).get('completed_stages', ()) else 1)" "%~1" "%~2" >nul 2>nul
exit /b %errorlevel%

:require_base_curriculum
%PYTHON% -c "import sys, torch; from masked_pretraining.masking import STAGES; checkpoint=torch.load(sys.argv[1], map_location='cpu', weights_only=False); completed=checkpoint.get('metadata', {}).get('completed_stages', ()); missing=[stage for stage in STAGES if stage not in completed]; assert not missing, 'missing base stages: ' + ', '.join(missing)" "%BASE_CHECKPOINT%"
exit /b %errorlevel%

:fail
echo Preliminary %MODEL% pipeline failed.
cd /d "%START_DIR%"
exit /b 1
