"""Keep the existing read-only observer in one bounded local process."""
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
        if worker.get("phase") != "huginn_fit" or worker.get("stop_requested"):
            print(output, flush=True)
            break
        watch_line = next(line[len("local_watch "):] for line in lines if line.startswith("local_watch "))
        watch = ast.literal_eval(watch_line)
        rows = [json.loads(line[line.index('{"fit_id":'):]) for line in lines if '{"fit_id":' in line]
        last = rows[-1] if rows else {}
        print(json.dumps({"at": worker["updated_utc"], "phase": worker["phase"],
            "child": worker.get("child_pid"), "n_done": last.get("n_done"),
            "last_paragraph_seconds": last.get("elapsed_seconds"),
            "gpu": next((line for line in lines if line.startswith("gpu ")), None),
            "combined_spend_upper": watch.get("observed_combined_spend_upper_usd"),
            "watch_last_poll": watch.get("last_poll_utc")}), flush=True)
    time.sleep(42)
