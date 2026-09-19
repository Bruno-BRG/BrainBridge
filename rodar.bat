@echo off
rem BrainBridge - roda o app desktop (Tauri dev: frontend + backend sidecar)
chcp 65001 >nul
cd /d "%~dp0webui"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERRO] Node.js nao encontrado no PATH. Instale em https://nodejs.org
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo [1/2] Instalando dependencias do frontend...
  call npm.cmd install
  if errorlevel 1 (
    echo [ERRO] Falha no npm install.
    pause
    exit /b 1
  )
)

echo [2/2] Iniciando BrainBridge (feche a janela do app para encerrar)...
call npx.cmd tauri dev
pause
