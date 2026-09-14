#!/usr/bin/env bash
# FALLBACK ONLY. Run by hand if the B300 pod's evaluation cannot finish before
# its provider TTL. Downloads the published 100-prompt lenses for run 0359 and
# runs the same evaluations run_b300.sh would have run, locally on the 5070 Ti.
# Mirrors run_b300.sh lines 755-763 and 810 (lens spec format: <target>=<path>).
# Does not touch the pod. Idempotent: skips downloads that already exist.
set -euo pipefail
cd /home/moloch/ouro_project
RUN=jlens-b300-20260905-0359
REPO=Vykos/ouro-jlens-results
LENS=artifacts/jlens/lens/b300_${RUN}
EVAL=artifacts/jlens/eval/b300_${RUN}
PY=venv/bin/python
mkdir -p "$LENS" "$EVAL"

download() {  # download <name>.pt and its .json sidecar; return 1 unless both end up present
  local name=$1 ext
  for ext in pt json; do
    [[ -f "$LENS/$name.$ext" ]] && continue
    "$PY" - "$REPO" "$RUN/artifacts/lens/n100/$name.$ext" "$LENS" <<'PY' || true
import shutil, sys
from huggingface_hub import hf_hub_download
repo, remote, dest = sys.argv[1:4]
tok = open("/home/moloch/Documents/Credentials/hf-token.txt").read().strip()
try:
    p = hf_hub_download(repo, remote, revision="main", token=tok)
except Exception as exc:
    sys.exit(f"missing on hub: {remote} ({type(exc).__name__})")
shutil.copy2(p, f"{dest}/{remote.split('/')[-1]}")
print("downloaded", remote)
PY
  done
  [[ -f "$LENS/$name.pt" && -f "$LENS/$name.json" ]]
}

# 1. exit-3 lenses (main + nested prefixes). Merged lenses exist on the pod since
# ~05:40 but as of 05:47 only the SHARDS are published. Try the merged files
# first; if absent, download the five shards and merge locally (nested prefixes
# are merges of the leading shards: n8 = 0-8, n32 = 0-32, n56 = 0-56, n80 = 0-80).
merged_ok=1
for n in exit3 exit3_n8 exit3_n32 exit3_n56 exit3_n80; do download "$n" || merged_ok=0; done
if [[ $merged_ok -ne 1 ]]; then
  S=("exit3_shard_0000_0008" "exit3_shard_0008_0032" "exit3_shard_0032_0056" "exit3_shard_0056_0080" "exit3_shard_0080_0100")
  for s in "${S[@]}"; do download "$s"; done
  merge() { local out=$1; shift; [[ -f "$out" ]] || $PY src/ouro_jlens/fit_lens.py merge --out "$out" "$@"; }
  merge "$LENS/exit3_n8.pt"  "$LENS/${S[0]}.pt"
  merge "$LENS/exit3_n32.pt" "$LENS/${S[0]}.pt" "$LENS/${S[1]}.pt"
  merge "$LENS/exit3_n56.pt" "$LENS/${S[0]}.pt" "$LENS/${S[1]}.pt" "$LENS/${S[2]}.pt"
  merge "$LENS/exit3_n80.pt" "$LENS/${S[0]}.pt" "$LENS/${S[1]}.pt" "$LENS/${S[2]}.pt" "$LENS/${S[3]}.pt"
  merge "$LENS/exit3.pt"     "$LENS/${S[0]}.pt" "$LENS/${S[1]}.pt" "$LENS/${S[2]}.pt" "$LENS/${S[3]}.pt" "$LENS/${S[4]}.pt"
fi

[[ "${FALLBACK_DOWNLOAD_ONLY:-0}" == 1 ]] && { echo "download/merge staged in $LENS; stopping before evaluation"; exit 0; }

# 2. evaluations that need only exit 3
$PY src/ouro_jlens/evaluate.py --lens "3=$LENS/exit3.pt" --out "$EVAL/n100_exit3"
$PY src/ouro_jlens/analyze.py --eval "$EVAL/n100_exit3"
for n in 8 32 56 80; do
  $PY src/ouro_jlens/evaluate.py --lens "3=$LENS/exit3_n$n.pt" --out "$EVAL/fitsize_n$n"
  $PY src/ouro_jlens/analyze.py --eval "$EVAL/fitsize_n$n"
done

# 3. all-exits evaluation, only if exits 0-2 were published (merged lenses exit0.pt, exit1.pt, exit2.pt)
if for n in exit0 exit1 exit2; do download "$n"; done; then
  $PY src/ouro_jlens/evaluate.py --lens "0=$LENS/exit0.pt" --lens "1=$LENS/exit1.pt" \
      --lens "2=$LENS/exit2.pt" --lens "3=$LENS/exit3.pt" --out "$EVAL/n100_allexits"
  $PY src/ouro_jlens/analyze.py --eval "$EVAL/n100_allexits"
fi
echo "done: $EVAL"
