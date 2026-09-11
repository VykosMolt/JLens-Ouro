"""Cost-only check for fitting the entire final pass, without task readouts."""
import json
import sys
import time
from pathlib import Path

import torch
from jlens.hooks import ActivationRecorder

sys.path.insert(0, "/home/moloch/ouro_project/src")
from ouro_jlens.bench import TEXT
from ouro_jlens.recurrent import load_ouro

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
model = load_ouro()
sources = list(range(144, 191))
results = []
for dim_batch in (8, 4, 2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        ids = model.encode(TEXT, max_length=128)
        with ActivationRecorder(model.layers, at=[*sources, 191], start_graph_at=144) as rec, torch.enable_grad():
            start = time.perf_counter()
            model.forward(ids.expand(dim_batch, -1))
            torch.cuda.synchronize()
            forward = time.perf_counter() - start
            target = rec.activations[191]
            cotangent = torch.zeros_like(target)
            for b in range(dim_batch):
                cotangent[b, 16:-1, b] = 1
            durations = []
            for _ in range(4):
                start = time.perf_counter()
                grads = torch.autograd.grad(target, [rec.activations[v] for v in sources], cotangent, retain_graph=True)
                rows = [g[:, 16:-1].float().mean(1).cpu() for g in grads]
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                del grads, rows
        result = {"dim_batch": dim_batch, "pass": True, "source_layers": sources,
                  "forward_seconds": forward, "backward_seconds": sum(durations[1:]) / 3,
                  "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        result["estimated_seconds_per_prompt"] = forward + result["backward_seconds"] * 2048 / dim_batch
        results.append(result)
        break
    except torch.OutOfMemoryError as error:
        results.append({"dim_batch": dim_batch, "pass": False, "error": str(error)})
        torch.cuda.empty_cache()
(HERE / "preflight_loop4.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2), flush=True)
