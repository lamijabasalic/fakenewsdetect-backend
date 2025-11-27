@echo off
echo Starting Backend Server...
cd /d %~dp0
C:\Users\Lamija\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause

