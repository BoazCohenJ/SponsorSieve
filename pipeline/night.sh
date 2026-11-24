#!/usr/bin/env bash
# Baseline sweep, retrain (best checkpoint by ad F1), export, sweep again.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
log() { echo "$1 $(date)" >> logs/night.log; }
log START
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen3.5:9b","keep_alive":0}' > /dev/null
$PY -u pipeline/sweep.py models/sieve-base model_int8.onnx > logs/sweep_base_int8.log 2>&1 || log "FAILED sweep base"
log "STAGE train"
$PY -u pipeline/train.py answerdotai/ModernBERT-base models/sieve-f1 3 > logs/train_f1.log 2>&1 || { log "FAILED train"; exit 1; }
log "STAGE export"
$PY -u pipeline/export_onnx.py models/sieve-f1 > logs/export_f1.log 2>&1 || { log "FAILED export"; exit 1; }
log "STAGE sweep"
$PY -u pipeline/sweep.py models/sieve-f1 model_int8.onnx > logs/sweep_f1_int8.log 2>&1 || log "FAILED sweep f1 int8"
$PY -u pipeline/sweep.py models/sieve-f1 model.onnx > logs/sweep_f1_fp32.log 2>&1 || log "FAILED sweep f1 fp32"
log DONE
