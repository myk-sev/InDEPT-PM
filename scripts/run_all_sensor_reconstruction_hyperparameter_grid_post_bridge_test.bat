@echo off
setlocal
set "REPO_ROOT=%~dp0.."
set "START_DIR=%CD%"
cd /d "%REPO_ROOT%" || exit /b 1

rem Runs one inference pass for every original reconstruction masking type on
rem each of the 72 final bridge checkpoints. Positive differences mean loss rose.
rem Usage: scripts\run_all_sensor_reconstruction_hyperparameter_grid_post_bridge_test.bat [options]
.venv\Scripts\python.exe -m masked_pretraining.post_bridge_test %*
set "EXIT_CODE=%ERRORLEVEL%"

cd /d "%START_DIR%"
exit /b %EXIT_CODE%
