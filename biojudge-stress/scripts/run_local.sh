#!/bin/bash
# Drive the local Ollama run in phases so every model runs fully on a 6 GB GPU:
#   A: padding/normaliser (Phi-4-mini)            4 slots
#   B: judging + selection by Llama, Qwen, Gemma   8 slots   (warms the shared SQLite cache)
#   C: full judge + select stages, all 4 judges    4 slots   (only Ministral calls are new)
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=runs/local/driver.log

start_server() {
  /d/ollama_models/serve.sh "$1" &
  SPID=$!
  until curl -s -m 2 http://127.0.0.1:11435/api/tags >/dev/null; do sleep 2; done
  echo "$(date) server up, parallel=$1" >> $LOG
}
stop_server() {
  taskkill //F //T //PID "$(cat /proc/$SPID/winpid)" >/dev/null 2>&1
  # llama-server runners can outlive their ollama parent; remove any that serve models from D:\ollama_models
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { \$_.CommandLine -like '*ollama_models*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
  sleep 3
  echo "$(date) server stopped" >> $LOG
}
phase() {  # name parallel config stages
  echo "$(date) phase $1 start" >> $LOG
  start_server "$2"
  $PY -m biojudge.run --config "$3" --stages "$4" >> runs/local/log_phase$1.txt 2>&1
  rc=$?
  stop_server
  echo "$(date) phase $1 exit $rc" >> $LOG
  [ $rc -eq 0 ] || exit $rc
}
phase A 4 configs/ollama_local.yaml perturb
phase B 8 configs/ollama_local_phaseB.yaml judge,select
phase C 4 configs/ollama_local.yaml judge,select
echo "$(date) ALL DONE" >> $LOG
