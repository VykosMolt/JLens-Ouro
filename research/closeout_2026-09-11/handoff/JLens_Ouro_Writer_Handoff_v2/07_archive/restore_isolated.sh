#!/usr/bin/env bash
# Run the restoration test with live project/cache paths hidden (empty bind mounts) in a private user+mount+network namespace.
set -euo pipefail
OUT=/tmp/claude-1000/RESTORATION_FROM_SSD.json
WORK=/tmp/claude-1000/jlens_restore_$$
mkdir -p "$WORK/empty"
unshare -Urm --net bash -c "
  set -e
  mount --bind $WORK/empty /home/moloch/ouro_project/jacobian-lens/research
  mount --bind $WORK/empty /home/moloch/ouro_project/artifacts
  mount --bind $WORK/empty /home/moloch/.cache/huggingface
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
  cp /tmp/claude-1000/jlens_restore_stage.py $WORK/restore_from_ssd.py; /home/moloch/ouro_project/venv/bin/python -B $WORK/restore_from_ssd.py $OUT $WORK/extract
"
