"""Read-only continuation monitor through evaluation and terminal validation."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

observer = Path(__file__).with_name("observe_attempt06.py")
failures = 0
for _ in range(300):
    result = subprocess.run([sys.executable, str(observer)], capture_output=True, text=True)
    output = result.stdout + result.stderr
    lines = output.splitlines()
    worker_line = next((line[7:] for line in lines if line.startswith("worker ")), None)
    worker = json.loads(worker_line) if worker_line is not None else None
    if result.returncode or worker is None:
        failures += 1
        print(json.dumps({"observation_failed": True, "consecutive_failures": failures,
                          "returncode": result.returncode, "output": output}), flush=True)
        if failures >= 3:
            break
    else:
        failures = 0
        if worker.get("phase") in ("complete", "failed", "stopped") or worker.get("stop_requested"):
            print(output, flush=True)
            break
        watch_line = next(line[len("local_watch "):] for line in lines if line.startswith("local_watch "))
        watch = ast.literal_eval(watch_line)
        progress = []
        for line in lines:
            for marker in ('{"stage": "huginn_readout"', '{"fit_id":'):
                if marker in line:
                    progress.append(json.loads(line[line.index(marker):]))
                    break
        print(json.dumps({"at": worker["updated_utc"], "phase": worker["phase"],
            "child": worker.get("child_pid"), "progress": progress[-1] if progress else None,
            "gpu": next((line for line in lines if line.startswith("gpu ")), None),
            "combined_spend_upper": watch.get("observed_combined_spend_upper_usd"),
            "watch_last_poll": watch.get("last_poll_utc")}), flush=True)
    time.sleep(42)
