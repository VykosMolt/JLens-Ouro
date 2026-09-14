"""End-of-run summary aid: which claims moved, from the frozen pre-run baseline to the
regenerated claim_status.json. Read-only. Prints a table; upgrades must be justified by
artifacts, never by this script."""
import json, sys
from pathlib import Path
pre = json.loads(Path("docs/jlens/watch/claim_status.prerun.json").read_text())
post = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/jlens/final/claim_status.json").read_text())
status = lambda v: v.get("status", v) if isinstance(v, dict) else v
keys = sorted(set(pre) | set(post))
w = max(map(len, keys))
print(f"{'claim':{w}}  {'pre-run':52}  post-run")
for k in keys:
    a, b = status(pre.get(k, "-")), status(post.get(k, "-"))
    mark = "  " if a == b else "->"
    print(f"{k:{w}}  {str(a):52} {mark} {b}")
