@echo off
REM Runs the daily batch training pipeline (preprocess -> train).
REM Point Windows Task Scheduler at this file, scheduled for ~23:30 daily.

REM --- EDIT THESE TWO LINES FOR YOUR MACHINE ---
set PROJECT_DIR=E:\SEM5\MLOPs\CP\Main\fire-anomaly-mlops
set CONDA_ENV=mlops

call conda activate %CONDA_ENV%
cd /d %PROJECT_DIR%
if not exist logs mkdir logs

echo [%date% %time%] Starting batch pipeline >> logs\batch_run.log
set RESAMPLE_FREQ=2min && python -m src.pipelines.preprocess >> logs\batch_run.log 2>&1
python -m src.pipelines.train >> logs\batch_run.log 2>&1
echo [%date% %time%] Batch pipeline finished >> logs\batch_run.log
