"""Execute the frozen experiment inside one externally supervised GPU lease."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request

import run_refits as io

HERE = Path(__file__).resolve().parent


def stamp():
    return datetime.now(timezone.utc).isoformat()


def download_model(manifest_path, destination):
    manifest = io._json(manifest_path)
    io._mkdir(destination)
    for record in manifest["files"]:
        path = io._no_links(destination / io._relative(record["path"]))
        if path.exists():
            io._verify_record(path, {key: record[key] for key in ("bytes", "sha256")})
            continue
        io._mkdir(path.parent)
        url = ("https://huggingface.co/" + manifest["repo_id"] + "/resolve/" + manifest["revision"]
               + "/" + urllib.parse.quote(record["path"], safe="/"))
        scratch = path.with_name(path.name + ".download")
        for attempt in range(3):
            try:
                digest, count = hashlib.sha256(), 0
                with urllib.request.urlopen(url, timeout=120) as response, scratch.open("wb") as output:
                    while block := response.read(8 * 1024 ** 2):
                        output.write(block)
                        digest.update(block)
                        count += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError("download differs from the frozen snapshot manifest")
                os.replace(scratch, path)
                io._fsync_dir(path.parent)
                break
            except Exception:
                scratch.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        print(json.dumps({"downloaded": record["path"], "bytes": record["bytes"]}), flush=True)
    io._snapshot(destination, manifest)


def projection(lease, profiles, *, now, huginn=None):
    """A timing estimate plus explicit allowances; the lease enforces the cap."""
    def valid_timing(row):
        values = [row["prompt_with_checkpoint_seconds"], row["optimized_seconds"],
                  row["io"]["accumulate_write_hash_prune_seconds"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("preflight timing is invalid")
        if not math.isclose(values[0], values[1] + values[2], rel_tol=1e-12):
            raise ValueError("preflight total omits compute or checkpoint time")
    if (len(profiles) != 3 or {row["name"] for row in profiles}
            != {"ouro_main", "ouro_penultimate", "ouro_positions"}):
        raise ValueError("preflight profile membership differs from the workload")
    for row in profiles:
        valid_timing(row)
    by_name = {row["name"]: row for row in profiles}
    if huginn is None:
        main = by_name["ouro_main"]
        ouro = sum(n * by_name[name]["prompt_with_checkpoint_seconds"] for name, n in
                   (("ouro_main", 500), ("ouro_penultimate", 100), ("ouro_positions", 100)))
        h_seconds = (main["optimized_seconds"] * 3.8363
                     + main["io"]["accumulate_write_hash_prune_seconds"] * (32 * 5280 ** 2) / (191 * 2048 ** 2))
        fit_seconds = ouro + 100 * h_seconds
        allowance = 3 * 3600  # Huginn gate, all evaluation, interpretation/checks.
        basis = "measured Ouro profiles; Huginn arithmetic proxy pending its own gate"
    else:
        if huginn.get("name") != "huginn_r8":
            raise ValueError("Huginn timing profile differs from the frozen geometry")
        valid_timing(huginn)
        h_seconds = huginn["prompt_with_checkpoint_seconds"]
        if not math.isfinite(h_seconds) or h_seconds <= 0:
            raise ValueError("Huginn preflight timing is invalid")
        fit_seconds, allowance = 100 * h_seconds, 3600
        basis = "measured complete Huginn B8 paragraph and checkpoint I/O"
    projected_finish = now + fit_seconds * 1.10 + allowance
    deadline = io._deadline(lease["work_deadline_utc"]).timestamp()
    projected_charge = (lease["prior_spend_upper_usd"]
                        + (projected_finish + 3600 - io._deadline(lease["billing_start_utc"]).timestamp())
                        / 3600 * lease["all_in_rate"])
    return {"schema_version": 1, "recorded_utc": stamp(), "basis": basis,
            "fit_seconds": fit_seconds, "timing_margin": 0.10, "other_work_allowance_seconds": allowance,
            "retrieval_allowance_seconds": 3600, "huginn_seconds_per_prompt": h_seconds,
            "projected_combined_cost_usd": projected_charge,
            "fits_before_work_deadline": projected_finish < deadline,
            "within_combined_budget": projected_charge < 24.85,
            "status": "passed" if projected_finish < deadline and projected_charge < 24.85 else "stopped",
            "limitation": "projection is not a runtime guarantee; fixed provider and watcher deadlines remain active"}


class Worker:
    def __init__(self, root, lease):
        self.root, self.lease = root, lease
        self.results = io._mkdir(root / "results")
        self.logs = io._mkdir(self.results / "logs")
        self.stop = False
        self.setup_complete = False
        self.child = None
        self.phase = "setup"
        self.python = str(root / "venv/bin/python")
        self.ouro_src = root / "bundle/ouro_project/src"
        self.ouro = root / "models/ouro"
        self.huginn = root / "models/huginn"
        self.deadline = io._deadline(lease["work_deadline_utc"]).timestamp()
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HOME": str(root / "runtime_cache/huggingface"),
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
        # No provider credential is needed by any GPU-side process.
        if any(key in self.env for key in ("RUNPOD_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
            raise ValueError("unexpected account credential in worker environment")

    def status(self, phase, **fields):
        self.phase = phase
        io._atomic_json(self.root / "status.json", {
            "schema_version": 1, "lease_name": self.lease["name"], "phase": phase,
            "updated_utc": stamp(), "setup_complete": self.setup_complete, **fields})

    def should_stop(self):
        return self.stop or (self.root / "STOP").exists() or time.time() >= self.deadline

    def run(self, name, command, *, timeout=None):
        if self.should_stop():
            raise InterruptedError("worker stop/deadline reached")
        self.status(name, command=[str(value) for value in command])
        started, sent = time.monotonic(), None
        with (self.logs / (name + ".log")).open("xb") as log:
            self.child = subprocess.Popen([str(value) for value in command], stdout=log,
                                          stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
            process_group = self.child.pid
            try:
                while self.child.poll() is None:
                    expired = timeout is not None and time.monotonic() - started >= timeout
                    if self.should_stop() or expired:
                        try:
                            if sent is None:
                                os.killpg(self.child.pid, signal.SIGTERM)
                                sent = time.monotonic()
                            elif time.monotonic() - sent > 90:
                                os.killpg(self.child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    self.status(name, elapsed_seconds=time.monotonic() - started,
                                child_pid=self.child.pid, stop_requested=sent is not None)
                    time.sleep(5)
                code = self.child.wait()
            finally:
                # A stage's parent can exit while a same-group descendant is
                # still writing. Keep the group identity until cleanup ends.
                def group_signal(sig):
                    try:
                        os.killpg(process_group, sig)
                    except ProcessLookupError:
                        return False
                    return True
                if group_signal(signal.SIGTERM):
                    cleanup_deadline = time.monotonic() + 10
                    while group_signal(0) and time.monotonic() < cleanup_deadline:
                        self.child.poll()
                        time.sleep(0.1)
                    group_signal(signal.SIGKILL)
                self.child.wait(timeout=10)
                self.child = None
        if sent is not None:
            raise InterruptedError("stage reached its stop/deadline")
        if code:
            raise RuntimeError(f"{name} exited with status {code}; see its result log")
        return time.monotonic() - started

    def script(self, name, filename, *args, timeout=None):
        return self.run(name, [self.python, HERE / filename, *args], timeout=timeout)

    def require_complete(self, *directories):
        for directory in directories:
            path = io._no_links(directory / "COMPLETE.json")
            if not path.exists():
                raise InterruptedError(f"{directory.relative_to(self.results)} stopped without a complete result")
            io._json(path)

    def validate_ouro(self):
        import run_huginn
        args = SimpleNamespace(run_spec=HERE / "run_spec.json", ouro_src=self.ouro_src,
                               combined_contract=HERE / "combined_contract.json",
                               calibration=HERE / "huginn_calibration.json",
                               ouro_evaluation_dir=self.results / "ouro_evaluation",
                               controls_evaluation_dir=self.results / "controls_evaluation")
        return run_huginn._evaluation_gate(args, run_huginn.contract(args))

    def interpretation(self):
        main = self.results / "ouro_evaluation"
        controls = self.results / "controls_evaluation"
        self.validate_ouro()
        expected = {"main_complete": io._record(main / "COMPLETE.json"),
                    "controls_complete": io._record(controls / "COMPLETE.json"),
                    "run_spec_sha256": io._file_hash(HERE / "run_spec.json"),
                    "combined_contract_sha256": io._file_hash(HERE / "combined_contract.json")}
        started = time.monotonic()
        while True:
            self.status("awaiting_ouro_interpretation", required_bindings=expected)
            path = self.root / "OURO_INTERPRETATION.json"
            if path.exists():
                value = io._json(path)
                if (value.get("status") != "interpreted" or value.get("lease_name") != self.lease["name"]
                        or value.get("bindings") != expected or not isinstance(value.get("findings"), str)
                        or len(value["findings"].strip()) < 100):
                    raise ValueError("Ouro interpretation receipt lacks matching results and findings")
                io._new_json(self.results / "OURO_INTERPRETATION.json", value)
                return
            if self.should_stop() or time.monotonic() - started > 3600:
                raise InterruptedError("Ouro interpretation was not received within its bounded window")
            time.sleep(15)

    def execute(self):
        for name in ("run_spec.json", "combined_contract.json", "environment.json", "huginn_eligibility.json"):
            with (self.results / name).open("xb") as handle:
                handle.write((HERE / name).read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
        self.run("venv", [sys.executable, "-m", "venv", self.root / "venv"], timeout=120)
        self.run("packages", [self.python, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--require-hashes",
                              "--only-binary=:all:", "-r", HERE / "requirements.lock"], timeout=2400)
        for model, destination, manifest in (("ouro", self.ouro, "model_manifest.json"),
                                              ("huginn", self.huginn, "huginn_model_manifest.json")):
            self.script("download_" + model, "cloud_worker.py", "--download-model", HERE / manifest,
                        "--destination", destination, timeout=2700)
        self.setup_complete = True
        common = ["--ouro-src", self.ouro_src]
        ouro_common = [*common, "--snapshot", self.ouro]
        stop = ["--stop-at-utc", self.lease["work_deadline_utc"]]
        self.script("preflight_ouro", "preflight_gpu.py", "--kind", "ouro", *ouro_common,
                    "--output-dir", self.results / "preflight_ouro", timeout=5400)
        gate = io._json(self.results / "preflight_ouro/COMPLETE.json")
        if gate.get("status") != "passed" or gate.get("kind") != "ouro":
            raise ValueError("Ouro GPU preflight did not pass")
        estimate = projection(self.lease, gate["profiles"], now=time.time())
        io._new_json(self.results / "initial_budget_projection.json", estimate)
        if estimate["status"] != "passed":
            raise InterruptedError("complete frozen workload does not fit the measured budget projection")
        self.script("ouro_fits", "run_refits.py", *ouro_common, "--all",
                    "--output-dir", self.results / "ouro", *stop)
        self.require_complete(*[self.results / f"ouro/fit_{index:02d}" for index in range(1, 6)])
        self.script("ouro_evaluation", "evaluate_refits.py", *ouro_common, "--fit-dir",
                    *[self.results / f"ouro/fit_{index:02d}" for index in range(1, 6)],
                    "--out", self.results / "ouro_evaluation", timeout=3600)
        self.require_complete(self.results / "ouro_evaluation")
        self.script("control_fits", "run_controls.py", *ouro_common,
                    "--main-output-dir", self.results / "ouro", "--output-dir", self.results / "controls", *stop)
        self.require_complete(self.results / "controls/ouro_penultimate/fit_01",
                              self.results / "controls/ouro_positions/fit_01")
        self.script("controls_evaluation", "evaluate_controls.py", *ouro_common,
                    "--main-evaluation-dir", self.results / "ouro_evaluation",
                    "--penultimate-fit-dir", self.results / "controls/ouro_penultimate/fit_01",
                    "--positions-fit-dir", self.results / "controls/ouro_positions/fit_01",
                    "--out", self.results / "controls_evaluation", timeout=3600)
        self.require_complete(self.results / "controls_evaluation")
        self.interpretation()
        huginn_common = [*common, "--snapshot", self.huginn,
                         "--ouro-evaluation-dir", self.results / "ouro_evaluation",
                         "--controls-evaluation-dir", self.results / "controls_evaluation"]
        self.script("preflight_huginn", "preflight_gpu.py", "--kind", "huginn", *huginn_common,
                    "--output-dir", self.results / "preflight_huginn", timeout=5400)
        hgate = io._json(self.results / "preflight_huginn/COMPLETE.json")
        if hgate.get("status") != "passed" or hgate.get("kind") != "huginn" or len(hgate["profiles"]) != 1:
            raise ValueError("Huginn GPU preflight did not pass")
        estimate = projection(self.lease, gate["profiles"], now=time.time(), huginn=hgate["profiles"][0])
        io._new_json(self.results / "huginn_budget_projection.json", estimate)
        if estimate["status"] != "passed":
            raise InterruptedError("Huginn N100 does not fit its measured remaining budget")
        self.script("huginn_fit", "run_huginn.py", *huginn_common,
                    "--run-spec", HERE / "run_spec.json", "--combined-contract", HERE / "combined_contract.json",
                    "--calibration", HERE / "huginn_calibration.json",
                    "--output-dir", self.results / "huginn", *stop)
        self.require_complete(self.results / "huginn/fit_01")
        self.script("huginn_evaluation", "evaluate_huginn.py", *common,
                    "--ouro-snapshot", self.ouro, "--huginn-snapshot", self.huginn,
                    "--fit-dir", self.results / "huginn/fit_01", "--out", self.results / "huginn_evaluation", *stop)
        self.require_complete(self.results / "huginn_evaluation")
        # Validate all three semantic completion seals again before claiming completion.
        self.validate_ouro()
        self.script("validate_huginn", "evaluate_huginn.py", "--validate-output", self.results / "huginn_evaluation",
                    timeout=600)
        io._new_json(self.results / "EXPERIMENT_COMPLETE.json", {
            "status": "complete", "lease_name": self.lease["name"], "completed_utc": stamp(),
            "evaluations": {name: io._record(self.results / name / "COMPLETE.json") for name in
                            ("ouro_evaluation", "controls_evaluation", "huginn_evaluation")}})
        self.status("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--download-model", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.download_model:
        if args.destination is None:
            parser.error("model download requires --destination")
        download_model(args.download_model, args.destination)
        return
    if args.root is None or args.lease is None:
        parser.error("worker requires --root and --lease")
    lease = io._json(args.lease)
    if lease["name"] != (args.root / "provider_lease_name").read_text().strip():
        raise ValueError("worker configuration differs from provider lease name")
    worker = Worker(args.root, lease)
    def stop_handler(*_):
        worker.stop = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    with io._output_lock(args.root / "worker_lock"):
        try:
            worker.execute()
        except InterruptedError as error:
            worker.status("stopped", reason=str(error))
        except Exception as error:
            worker.status("failed", error_type=type(error).__name__, reason=str(error))
            raise


if __name__ == "__main__":
    main()
