#!/usr/bin/env bash
# Transcribe, then label, train, export and evaluate. Logs in logs/.
set -u
cd "$(dirname "$0")/.."
echo "STAGE transcribe $(date)"
uv run python -u pipeline/transcribe.py >> logs/transcribe.log 2>&1
echo "STAGE label $(date)"
uv run python -u pipeline/label.py run > logs/label.log 2>&1 || { echo "FAILED label"; exit 1; }
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen3.5:9b","keep_alive":0}' >/dev/null
echo "STAGE train $(date)"
uv run python -u pipeline/train.py answerdotai/ModernBERT-base models/sieve-base 3 > logs/train.log 2>&1 || { echo "FAILED train"; exit 1; }
echo "STAGE export $(date)"
uv run python -u pipeline/export_onnx.py models/sieve-base > logs/export.log 2>&1 || { echo "FAILED export"; exit 1; }
echo "STAGE evaluate $(date)"
uv run python -u pipeline/evaluate.py models/sieve-base model.onnx > logs/eval_fp32.log 2>&1
uv run python -u pipeline/evaluate.py models/sieve-base model_int8.onnx > logs/eval_int8.log 2>&1
echo "DONE $(date)"
tail -1 logs/eval_fp32.log logs/eval_int8.log
