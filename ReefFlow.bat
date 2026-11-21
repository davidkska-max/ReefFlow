@echo off
rem Starts the ReefFlow browser interface (runs locally; no internet needed).
cd /d "%~dp0"
".venv\Scripts\reefflow.exe" ui %*
