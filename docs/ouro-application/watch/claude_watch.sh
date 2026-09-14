#!/usr/bin/env bash
# CLAUDE's independent watcher. Emits one line per poll; every line is an event that
# wakes the session. Appends HEARTBEAT/ANOMALY entries to the board. Read-only except
# for the board file. Never creates or runs a pod.
set -u
cd /home/moloch/ouro_project
BOARD=docs/jlens/watch/BOARD.md
SHA=$(cat /tmp/claude-1000/-home-moloch-ouro-project/6c2195c9-8eae-43c2-bdf6-39cd670c7b37/scratchpad/watch_key.sha256)
INTERVAL=${1:-300}      # seconds between polls (5 min)
STALE=1500              # other watcher stale after 25 min
post() { printf '\n## %s CLAUDE %s\n%s\n' "$(date +%H:%M:%S)" "$1" "$2" >> "$BOARD"; }
last_hb=0; last_idle=0; prev=''; first_live=0
while true; do
  now=$(date +%s)
  st=$(venv/bin/python src/ouro_jlens/pod.py status --runpod-key-sha256 "$SHA" 2>&1 | tail -8)
  bal=$(grep -oE 'balance \$-?[0-9.]+' <<<"$st" | head -1); pods=$(grep -cE '^\s+[a-z0-9]{12,}\s' <<<"$st" || true)
  lease=""; for f in artifacts/jlens/runs/*/state.json; do [ -f "$f" ] && lease="$lease $(python3 -c "import json,sys;d=json.load(open('$f'));print(f\"{d.get('run_id','?')}:{d.get('status','?')}:pod={d.get('pod_id','-')}\")" 2>/dev/null)"; done
  unit=$(systemctl --user list-units --no-legend 2>/dev/null | grep -iE 'jlens|ouro' | awk '{print $1":"$3}' | tr '\n' ' ')
  line="$bal pods=$pods lease=[${lease# }] monitor=[${unit}]"
  # anomalies: balance under floor, pod present with no active lease, active lease but monitor unit not active
  anom=""
  b=$(grep -oE '\-?[0-9.]+' <<<"$bal" | head -1); [ -n "$b" ] && awk "BEGIN{exit !($b < 5.0)}" && anom="$anom BALANCE_BELOW_FLOOR"
  [ "$pods" -gt 0 ] && ! grep -qE ':(pending|active|monitoring|terminating)' <<<"$lease" && anom="$anom POD_WITHOUT_ACTIVE_LEASE"
  grep -qE ':(pending|active|monitoring|terminating)' <<<"$lease" && ! grep -q ':active' <<<"$unit" && anom="$anom LEASE_WITHOUT_LIVE_MONITOR"
  # Two-watcher check. A missing SOL heartbeat is as much an anomaly as a stale one;
  # give SOL 15 min after a pod first appears before flagging absence.
  sol=$(grep -E '^## [0-9:]{8} SOL HEARTBEAT' "$BOARD" | tail -1 | awk '{print $2}')
  if [ "$pods" -gt 0 ]; then
    [ "$first_live" -eq 0 ] && first_live=$now
    if [ -n "$sol" ]; then
      sol_s=$(date -d "$sol" +%s 2>/dev/null || echo 0); [ $((now - sol_s)) -gt $STALE ] && anom="$anom SOL_HEARTBEAT_STALE(${sol})"
    elif [ $((now - first_live)) -gt 900 ]; then anom="$anom SOL_NO_HEARTBEAT_SINCE_LAUNCH"; fi
  else first_live=0; fi
  # Idle (no pod, no lease): stay quiet except one liveness line per hour, so the
  # session is woken only by change. Live: every poll is an event.
  if [ -n "$anom" ]; then echo "ANOMALY$anom | $line"; post ANOMALY "$anom | $line"
  elif [ "$pods" -gt 0 ] || [ -n "$lease" ] || [ "$line" != "$prev" ] || [ $((now - last_idle)) -ge 3600 ]; then
    echo "OK | $line"; [ "$pods" -eq 0 ] && last_idle=$now; fi
  prev="$line"
  if [ "$pods" -gt 0 ] && [ $((now - last_hb)) -ge 600 ]; then post HEARTBEAT "$line"; last_hb=$now; fi
  sleep "$INTERVAL"
done
