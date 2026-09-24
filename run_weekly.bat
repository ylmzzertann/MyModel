@echo off
REM futbol-ml-modeli - haftalik otomatik guncelleme.
REM Windows Gorev Zamanlayici bu dosyayi calistirir ("futbol-ml-modeli haftalik guncelleme" gorevi).
REM Elle calistirmak icin: run_weekly.bat   (ek secenekler: --force-retrain, --no-retrain)
REM Sonuc raporu: reports\haftalik_rapor.md   Loglar: logs\
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m src.weekly_update %*
exit /b %ERRORLEVEL%
