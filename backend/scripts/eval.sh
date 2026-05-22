#!/usr/bin/env bash
set -euo pipefail

DOC_ID="${DOC_ID:-3}"
QA="${QA:-../data/qa_alpha_syn.jsonl}"
TOP_K="${TOP_K:-4}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="../reports/${TS}_doc${DOC_ID}"
mkdir -p "$OUT_DIR"

echo "Running retrieval eval..."
python3 -m app.eval.run --document_id "$DOC_ID" --qa "$QA" --top_k "$TOP_K" --skip_unanswerable \
  | tee "${OUT_DIR}/retrieval.json"

echo "Running generate eval..."
python3 -m app.eval.run_generate --document_id "$DOC_ID" --qa "$QA" --top_k "$TOP_K" \
  | tee "${OUT_DIR}/generate.json"

echo "Done. Reports saved to: ${OUT_DIR}"