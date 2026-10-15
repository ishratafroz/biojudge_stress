#!/usr/bin/env bash
# Ollama server for the judging phases on the Linux L40S host (2026-09-26).
# Runs on port 11436: 11434 is the user's own Ollama and 11435 is their OpenBioLLM adapter server.
# All four judge models stay resident (MAX_LOADED_MODELS=4), so the model-major job
# ordering in pipeline.py is no longer needed to avoid swapping.
set -euo pipefail
export OLLAMA_HOST=127.0.0.1:11436
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3}
export OLLAMA_MAX_LOADED_MODELS=4
export OLLAMA_NUM_PARALLEL=${1:-8}
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_KEEP_ALIVE=24h
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NOPRUNE=1
# llama-server's host prompt cache saves 50-70 MiB of slot state per idle slot; it burns time and
# RAM for no benefit here (every prompt is distinct). Same fix as on the RTX 3050 host.
export LLAMA_ARG_CACHE_RAM=0
exec "$HOME/ollama/bin/ollama" serve
