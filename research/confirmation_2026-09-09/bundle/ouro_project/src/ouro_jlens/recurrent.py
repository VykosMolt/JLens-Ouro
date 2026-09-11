"""Jacobian lens over Ouro's recurrent depth.

Ouro runs its shared decoder layers `total_ut_steps` times. jlens's
ActivationRecorder keys activations by the index it hooked, so on the native
forward every recurrent pass overwrites the previous one and only the last pass
survives. Here `model.layers` is a flat list of 192 virtual blocks indexed
`ut * 48 + layer`; each is a LoopTap that registers its hook on the shared
physical module but only fires when Ouro's own `current_ut` kwarg matches.
Everything else in jlens (fit / apply / merge) runs unmodified over virtual
indices.

Indexing: `ut` is Ouro's 0-based recurrent step (== `exit_at_step`), `layer`
is the 0-based physical decoder index. Human-facing "loop k" == ut k-1.
"""

from __future__ import annotations

from pathlib import Path

import torch
import transformers

import jlens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OURO_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
OURO_SNAPSHOT = (
    PROJECT_ROOT / "artifacts" / "hf_cache" / "hub" / "models--ByteDance--Ouro-2.6B"
    / "snapshots" / OURO_REVISION
)


def model_snapshot_files(path: str | Path = OURO_SNAPSHOT) -> list[Path]:
    """Return every regular file that can contribute to a local model load.

    Transformers may consult remote-code, tokenizer, generation, and shard
    metadata in addition to the primary config and weights.  Binding a
    hand-picked subset therefore does not identify the loaded checkpoint.
    The pinned Hugging Face snapshot is small apart from its weights, so the
    conservative and stable contract is every file below the snapshot.
    """

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"model snapshot root is missing or linked: {root}")
    files: list[Path] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root)
        current = candidate.parent
        while current != root:
            if current.is_symlink():
                raise ValueError(f"model snapshot traverses a linked directory: {relative}")
            current = current.parent
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_symlink():
            raise ValueError(f"model snapshot contains a dangling or directory link: {relative}")
    required = {"config.json", "model.safetensors", "modeling_ouro.py", "tokenizer.json"}
    present = {candidate.relative_to(root).as_posix() for candidate in files}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"model snapshot is missing required files: {', '.join(missing)}")
    return files


class LoopTap:
    """Hook target for one (recurrent step, physical layer) pair."""

    def __init__(self, block: torch.nn.Module, ut: int) -> None:
        self.block = block
        self.ut = ut

    def register_forward_hook(self, hook):
        def filtered(module, args, kwargs, output):
            if kwargs["current_ut"] == self.ut:
                hook(module, args, output)

        return self.block.register_forward_hook(filtered, with_kwargs=True)


class OuroLensModel(jlens.HFLensModel):
    def __init__(self, hf_model, tokenizer) -> None:
        super().__init__(hf_model, tokenizer, force_bos=False)
        self.hf_model = hf_model
        self.blocks = self.layers
        self.n_physical = len(self.blocks)
        self.n_ut = hf_model.model.total_ut_steps
        self.layers = [
            LoopTap(block, ut) for ut in range(self.n_ut) for block in self.blocks
        ]
        self.n_layers = len(self.layers)

    def index(self, ut: int, layer: int) -> int:
        if isinstance(ut, bool) or not isinstance(ut, int) or not 0 <= ut < self.n_ut:
            raise ValueError(f"recurrent step is outside [0, {self.n_ut}): {ut!r}")
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < self.n_physical:
            raise ValueError(f"physical layer is outside [0, {self.n_physical}): {layer!r}")
        return ut * self.n_physical + layer

    def split(self, virtual: int) -> tuple[int, int]:
        if (
            isinstance(virtual, bool)
            or not isinstance(virtual, int)
            or not 0 <= virtual < self.n_layers
        ):
            raise ValueError(
                f"virtual layer is outside [0, {self.n_layers}): {virtual!r}"
            )
        return divmod(virtual, self.n_physical)

    def exit_index(self, ut: int) -> int:
        return self.index(ut, self.n_physical - 1)

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.tokenizer(text, truncation=True, max_length=max_length - 1).input_ids
        ids = [self.tokenizer.bos_token_id, *ids]
        return torch.tensor([ids], device=self.input_device)


def load_ouro(path: str | Path = OURO_SNAPSHOT, device: str = "cuda", dtype=torch.bfloat16):
    snapshot = Path(path)
    path_string = str(snapshot)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path_string, trust_remote_code=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        path_string, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    wrapped = OuroLensModel(hf_model, tokenizer)
    # Downstream evidence must describe the checkpoint actually loaded.  The
    # previous wrapper silently left every alternate checkpoint looking like
    # the base Ouro revision to provenance code.
    wrapped.snapshot_path = snapshot.absolute()
    name = snapshot.name
    wrapped.model_revision = (
        name
        if len(name) == 40 and all(character in "0123456789abcdef" for character in name)
        else "LOCAL_UNREVISIONED"
    )
    return wrapped
