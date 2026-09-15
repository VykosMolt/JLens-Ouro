#!/usr/bin/env bash
# Wait for the shard download, then merge and evaluate. Logs: merge.log, evaluate.log. No token is used here.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/moloch/ouro_project/venv/bin/python
CAND="/home/moloch/ouro_project/jacobian-lens/research/refit_round_2026-09-07/cloud_leases/attempt_06/prefetch_v3_20260909T134030Z_649dff12/staging/ouro/fit_01/final/cursor_000100_e090a066bc3b4630891f195e7e7d1d29/lens.pt"
if [ "$(sha256sum "$CAND" | cut -c1-12)" = "90f01f6afd83" ]; then export FIT01_BANK="$CAND"; echo "fit01 bank verified: $CAND"; else echo "fit01 bank not verified; fit01 check skipped"; fi
while [ ! -f RETRIEVAL_RECEIPT.json ]; do sleep 30; done
echo "retrieval finished: $(tail -1 retrieval.log)"
$PY -B merge_shards.py > merge.log 2>&1 || { echo "MERGE FAILED"; tail -5 merge.log; exit 1; }
tail -6 merge.log
$PY -B evaluate_local_exit.py > evaluate.log 2>&1 || { echo "EVALUATION FAILED"; tail -15 evaluate.log; exit 1; }
tail -12 evaluate.log
