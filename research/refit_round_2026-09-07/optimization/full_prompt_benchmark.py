"""Complete upstream baseline on the fixed cost fixture; no task scoring."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import TEXT
from ouro_jlens.recurrent import load_ouro
import jlens.fitting


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    here = Path(__file__).resolve().parent
    model = load_ouro()
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": __doc__, "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "model_revision": model.model_revision,
        "prompt_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
        "fitting_source_sha256": hashlib.sha256(Path(jlens.fitting.__file__).read_bytes()).hexdigest(),
        "dim_batch": 2, "target": 191, "source_count": 191,
        "calibration_count_in_research": 0,
    }
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    maps, length, n_valid = jlens.fitting.jacobian_for_prompt(
        model, TEXT, list(range(191)), target_layer=191, dim_batch=2,
        max_seq_len=128, skip_first=16,
    )
    torch.cuda.synchronize()
    result["complete_prompt_seconds"] = time.perf_counter() - t0
    result["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
    result["sequence_length"] = length
    result["n_valid"] = n_valid
    result["all_finite"] = all(torch.isfinite(matrix).all().item() for matrix in maps.values())
    t0 = time.perf_counter()
    destination = here / "baseline_full.pt"
    torch.save(maps, destination)
    result["save_seconds"] = time.perf_counter() - t0
    result["artifact_bytes"] = destination.stat().st_size
    (here / "baseline_full.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
