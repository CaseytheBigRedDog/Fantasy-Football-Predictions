@echo off
REM Runs the weekly data refresh + full model retrain.
REM This file is meant to be pointed at by Windows Task Scheduler.
cd /d "%~dp0"
python refresh_data.py >> refresh_log.txt 2>&1
