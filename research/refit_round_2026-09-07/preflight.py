"""Measure the retained-graph fitter before choosing the independent-fit budget."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import bench
from ouro_jlens.recurrent import load_ouro


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "purpose": "cost/gradient feasibility only; no task-recovery evaluation",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "configurations": []}
    assert torch.cuda.is_available(), "CUDA is required for this preflight"
    result["device"] = torch.cuda.get_device_name(0)
    model = load_ouro()
    result["load_seconds"] = time.perf_counter() - start
    result["model_revision"] = model.model_revision
    result["model_training"] = model.hf_model.training
    result["parameter_gradients_enabled"] = any(p.requires_grad for p in model.hf_model.parameters())
    for batch in (4, 2, 1):
        try:
            measured = bench(model, dim_batch=batch, seq_len=128, n_passes=4)
            result["configurations"].append(measured)
            break
        except torch.OutOfMemoryError as error:
            result["configurations"].append({"dim_batch": batch, "pass": False,
                                              "error": str(error)})
            torch.cuda.empty_cache()
    result["elapsed_seconds"] = time.perf_counter() - start
    (HERE / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert any(x["pass"] for x in result["configurations"]), "No fitting configuration fits"


if __name__ == "__main__":
    main()
