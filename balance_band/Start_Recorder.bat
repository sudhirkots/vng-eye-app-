@echo off
REM Balance Band recorder: starts the local server and opens the recording page in the browser.
REM Use Google Chrome or Microsoft Edge (they can talk to the band over Bluetooth).
REM Needs Python 3 with numpy. Recordings are saved in %USERPROFILE%\BalanceBand\sessions
cd /d "%~dp0"
py -3 recorder\server.py %*
if errorlevel 9009 python recorder\server.py %*
pause
