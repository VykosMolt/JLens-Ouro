#!/usr/bin/env bash
# CLAUDE's fast progress watcher for ONE lease (~45s). Read-only except the board.
# Emits a line on any change of lease status / monitor unit / receipt set, plus a
# heartbeat every 180s; emits a TERMINAL line and exits when the lease ends.
# Never creates, touches, or terminates a pod.
cd "$(dirname "$0")/../../.." || exit 1
L=$1; RESULTS=$2; RUN=$3; BOARD=docs/jlens/watch/BOARD.md
post() { printf '\n## %s CLAUDE %s\n%s\n' "$(date +%H:%M:%S)" "$1" "$2" >> "$BOARD"; }
prev=""; last_hb=0; errs=0
while true; do
  now=$(date +%s)
  read -r status spend alive < <(venv/bin/python - "$L" <<'PY' 2>/dev/null || echo "state-read-error 0 0"
import json,sys,time,datetime as dt
d=json.load(open(f"artifacts/jlens/runs/{sys.argv[1]}/state.json"))
t=dt.datetime.fromisoformat(d["provider_terminate_after"].replace("Z","+00:00")).timestamp()-12600
print(d.get("status"), "%.2f"%(d.get("estimated_spend") or 0), "%d"%(time.time()-t))
PY
)
  unit=$(systemctl --user is-active "ouro-jlens-monitor-$L.service" 2>/dev/null || echo unknown)
  rec=$(venv/bin/python - "$RESULTS" "$RUN" <<'PY' 2>/dev/null || echo "hf-list-error"
import sys
from huggingface_hub import HfApi
tok=open("/home/moloch/Documents/Credentials/hf-token.txt").read().strip()
try:
    fs=sorted(t.path for t in HfApi(token=tok).list_repo_tree(sys.argv[1], path_in_repo=sys.argv[2], revision="main", recursive=True))
    print(f"{len(fs)}:{fs[-1].split('/')[-1][:60] if fs else '-'}")
except Exception as e:
    print("0:-" if "404" in str(e) or "Entry" in type(e).__name__ or "Revision" in type(e).__name__ else "hf-list-error")
PY
)
  [ "$rec" = "hf-list-error" ] && errs=$((errs+1)) || errs=0
  key="$status|$unit|$rec"
  line="lease=$status unit=$unit alive=${alive}s spend=\$$spend receipts=$rec"
  if [ "$key" != "$prev" ]; then
    echo "$(date +%T) CHANGE $line"
    [ -n "$prev" ] && [ "${rec%%:*}" != "${prev##*|}" ] && [ "$rec" != "hf-list-error" ] && post PROGRESS "$line"
    last_hb=$now
  elif (( now - last_hb >= 180 )); then
    echo "$(date +%T) HB $line"; last_hb=$now
  fi
  (( errs == 4 )) && echo "$(date +%T) WARN hf listing failed 4x in a row"
  if [ "$status" = "terminated" ] || [ "$unit" = "inactive" ] || [ "$unit" = "failed" ]; then
    echo "$(date +%T) TERMINAL $line"; post NOTE "fast watcher: lease reached terminal state — $line"; exit 0
  fi
  prev=$key; sleep 45
done
