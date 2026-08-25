@echo off
setlocal
set "REPO_ROOT=%~dp0.."

rem Usage: scripts\run_base_reconstruction_model_family.bat [device] [epochs_per_stage] [batch_size] [workers]
rem Add --dry-run first to print the 22-run matrix without starting training.
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
)

set "DEVICE=%~1"
set "EPOCHS_PER_STAGE=%~2"
set "BATCH_SIZE=%~3"
set "WORKERS=%~4"
if not defined DEVICE set "DEVICE=auto"
if not defined EPOCHS_PER_STAGE set "EPOCHS_PER_STAGE=20"
if not defined BATCH_SIZE set "BATCH_SIZE=64"
if not defined WORKERS set "WORKERS=0"

set "START_DIR=%CD%"
cd /d "%REPO_ROOT%" || exit /b 1

set "PYTHON=.venv\Scripts\python.exe"
set "OUTPUT_ROOT=masked_pretraining\runs\base_reconstruction"
set "ALL_DATA=inputs\reconstruction\all_sensors_exclusion_informed_finetuned_masked_training_data.csv"
set "K12_DATA=inputs\reconstruction\k12_exclusion_informed_finetuned_masked_training_data.csv"

for %%P in ("%PYTHON%" "%ALL_DATA%" "%K12_DATA%") do (
    if not exist "%%~P" (
        echo Required path not found: %%~P
        goto :fail
    )
)

set "TRAINING_COUNT=0"
call :run_dataset "all_excl_final" "%ALL_DATA%" || goto :fail
call :run_dataset "k12_excl_final" "%K12_DATA%" || goto :fail

echo Completed %TRAINING_COUNT% base-reconstruction training runs.
cd /d "%START_DIR%"
exit /b 0

:run_dataset
set "DATASET=%~1"
set "TRAINING_DATA=%~2"
set "AUDIT_DIR=%OUTPUT_ROOT%\%DATASET%\audit"
echo Dataset: %DATASET%
echo Training data: %TRAINING_DATA%
if not defined DRY_RUN (
    if not exist "%AUDIT_DIR%" mkdir "%AUDIT_DIR%" || exit /b 1
    %PYTHON% -m masked_pretraining audit ^
        --training-data "%TRAINING_DATA%" ^
        --output "%AUDIT_DIR%\dataset_audit.json" >nul
    if errorlevel 1 exit /b 1
)

for /f "delims=" %%M in ('%PYTHON% -c "from masked_pretraining.models import model_names; print(*model_names(), sep='\n')"') do (
    call :run_model "%%M" || exit /b 1
)
exit /b 0

:run_model
set "MODEL=%~1"
set "RUN_ROOT=%OUTPUT_ROOT%\%DATASET%\%MODEL%"
set /a TRAINING_COUNT+=1 >nul
echo.
echo Starting %DATASET% / %MODEL%
echo Artifacts: %RUN_ROOT%
if defined DRY_RUN exit /b 0

%PYTHON% -m masked_pretraining train ^
    --training-data "%TRAINING_DATA%" ^
    --model "%MODEL%" ^
    --stages points short_blocks mixed_blocks cross_channel suffix_3 suffix_6 suffix_12 ^
    --epochs-per-stage "%EPOCHS_PER_STAGE%" ^
    --patience "%EPOCHS_PER_STAGE%" ^
    --reconstruction-every-epochs 5 ^
    --batch-size "%BATCH_SIZE%" ^
    --workers "%WORKERS%" ^
    --device "%DEVICE%" ^
    --checkpoint "%RUN_ROOT%\checkpoints\run.pt"
if errorlevel 1 (
    echo Training failed for %DATASET% / %MODEL%.
    exit /b 1
)
exit /b 0

:fail
cd /d "%START_DIR%"
exit /b 1
