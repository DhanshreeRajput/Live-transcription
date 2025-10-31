@echo off
title Live Call Center Transcription Server
color 0A

echo ============================================================
echo  LIVE CALL CENTER TRANSCRIPTION - WHISPER LARGE-V3
echo ============================================================
echo.
echo  Starting server...
echo  Please wait while the model loads (first time: ~3GB download)
echo.
echo  Once you see "Server starting at http://localhost:8000"
echo  Open live.html in your browser
echo.
echo ============================================================
echo.

REM Start the server
python live_transcription_server.py

echo.
echo ============================================================
echo  Server stopped
echo ============================================================
pause