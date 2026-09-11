"""Gradient-capable R=8 J-Lens adapter for the pinned native Huginn model.

Virtual sources 0..31 are the four native core-block outputs at each of eight
passes. Coda outputs are 32 and 33; target 33 precedes the final norm. ``forward``
returns that target without computing vocabulary logits. ``unembed`` applies
the final native norm/head; ``coda_readout`` applies the trained coda to a FULL
raw core sequence, including the pre-coda norm and original positions.

One deterministic state is drawn per encoded prompt in the native dtype and
device, then copied identically into every derivative lane. The seed recipe
includes an explicit namespace: calibration and evaluation should use separate
namespaces/seeds. No global random generator is advanced by this adapter.

The loader accepts a local snapshot only. It checks the pinned code/config/
tokenizer/index bytes; its caller must separately verify a full weight manifest
before a scientific run. ``--self-test --snapshot PATH`` imports native code and
the tokenizer, then tests tiny CPU instances; it never loads checkpoint weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import types

import torch


REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
FIT_STATE_SEED = 2026090801
PINNED_SMALL_FILES = {
    "config.json": "e9fe79df06a783ca33a76038c59a715b6a62f15c4a5b17b9681e697fea46c79c",
    "raven_config_minimal.py": "36a39939e76dc9e8f7563187c6498e012ca7f64a3d86b32ff1daa7f4c41a4278",
    "raven_modeling_minimal.py": "a1d447da93c605a6dc11fbc17cbc7185663f4a41ae12335b34b5847edf0f84aa",
    "model.safetensors.index.json": "a28061573e97ba3645b482c11d82ba903b40942de0113472b30afb5a3cf4c1eb",
    "tokenizer.json": "9cc201a5061b70aba0d227ef9766fcfe21e0989e0c0ba442b4d7732c4de12308",
    "tokenizer_config.json": "ff9e171e16f200e7865c27fa38e661722848ab665af1610d5bb2b62502d6cdcf",
    "special_tokens_map.json": "1291ad5fcb7b7628c3885a2d973f165d006cb87530ca368db90374dc9942224e",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def verify_pinned_metadata(snapshot):
    """Verify small load inputs, not the contents of the four weight shards."""
    snapshot = Path(snapshot).absolute()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("snapshot must be an explicit local directory")
    for name, expected in PINNED_SMALL_FILES.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Huginn pinned metadata mismatch: {name}")
    return snapshot


def _native_classes(snapshot):
    """Load verified native Python sources without writing into the snapshot."""
    snapshot = verify_pinned_metadata(snapshot)
    suffix = hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
    package_name = f"_jlens_huginn_{suffix}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(snapshot)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        try:
            for stem in ("raven_config_minimal", "raven_modeling_minimal"):
                path = snapshot / f"{stem}.py"
                name = f"{package_name}.{stem}"
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                # compile/exec avoids a source-cache write in the read-only
                # checkpoint. These exact source bytes were checked above.
                exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        except BaseException:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    del sys.modules[name]
            raise
    return (sys.modules[f"{package_name}.raven_config_minimal"].RavenConfig,
            sys.modules[f"{package_name}.raven_modeling_minimal"].RavenForCausalLM)


class LoopTap:
    """A hook boundary selected by the block's actual native ``step_idx``."""

    def __init__(self, block, step_idx):
        self.block = block
        self.step_idx = step_idx

    def register_forward_hook(self, hook):
        def filtered(module, args, kwargs, output):
            step = kwargs.get("step_idx", args[2] if len(args) > 2 else None)
            if isinstance(step, torch.Tensor):
                if step.device.type != "cpu" or step.ndim != 0 or step.dtype != torch.long:
                    raise ValueError("native step_idx must be a scalar CPU int64 tensor")
                step = int(step.item())
            elif type(step) is not int:
                raise ValueError("native step_idx is missing or not an integer")
            if step == self.step_idx:
                # Preserve the normal PyTorch replacement-hook contract.
                return hook(module, args, output)
            return None

        return self.block.register_forward_hook(filtered, with_kwargs=True)


class HuginnLensModel:
    """A LensModel over the exact native R=8 recurrent/coda topology.

    ``cpu_test`` permits a tiny width/vocabulary, with the same native classes
    and 2/4/2 physical topology. It is never enabled by ``load_huginn``.
    """

    def __init__(self, native_model, tokenizer, *, state_seed=FIT_STATE_SEED,
                 seed_namespace="calibration", cpu_test=False):
        _integer("state_seed", state_seed)
        if state_seed >= 2 ** 63:
            raise ValueError("state_seed must fit in 63 nonnegative bits")
        if not isinstance(seed_namespace, str) or not seed_namespace:
            raise ValueError("seed_namespace must be a nonempty string")
        if type(cpu_test) is not bool:
            raise ValueError("cpu_test must be boolean")
        source_path = Path(inspect.getfile(type(native_model)))
        if (type(native_model).__name__ != "RavenForCausalLM"
                or hashlib.sha256(source_path.read_bytes()).hexdigest()
                != PINNED_SMALL_FILES["raven_modeling_minimal.py"]):
            raise ValueError("adapter requires the pinned native RavenForCausalLM class")
        self.hf_model = self._hf_model = native_model
        self.tokenizer = tokenizer
        self.state_seed = state_seed
        self.seed_namespace = seed_namespace
        self.n_ut = 8
        self.n_physical = 4
        self.n_prelude = 2
        self.n_coda = 2
        self.d_model = native_model.config.n_embd
        cfg = native_model.config
        stack = native_model.transformer
        if (len(stack.prelude), len(stack.core_block), len(stack.coda)) != (2, 4, 2):
            raise ValueError("Huginn adapter requires the native 2/4/2 block topology")
        if (cfg.test_time_noise != 0 or cfg.state_init != "like-init"
                or cfg.injection_type != "linear" or not cfg.tie_embeddings):
            raise ValueError("Huginn state/noise/injection configuration differs from the pinned contract")
        if any(hasattr(module, "_orig_mod") for module in native_model.modules()):
            raise ValueError("compiled modules are outside the adapter contract")
        if (not cpu_test and (cfg.n_embd, cfg.num_attention_heads, cfg.num_key_value_heads,
                              cfg.intermediate_size, cfg.padded_vocab_size) != (5280, 55, 55, 17920, 65536)):
            raise ValueError("production Huginn geometry does not match the pinned checkpoint")
        if cpu_test and any(parameter.device.type != "cpu" for parameter in native_model.parameters()):
            raise ValueError("cpu_test cannot wrap accelerator parameters")
        native_model.eval()
        native_model.requires_grad_(False)
        native_model.gradient_checkpointing = False
        parameters = list(native_model.parameters())
        if (not parameters or any(parameter.dtype != parameters[0].dtype
                                  or parameter.device != parameters[0].device for parameter in parameters)
                or parameters[0].dtype not in (torch.float32, torch.bfloat16, torch.float16)):
            raise ValueError("native parameters must share one floating dtype and one device")
        if not math.isclose(cfg.init_values["std"], math.sqrt(2 / (5 * self.d_model)), rel_tol=1e-14):
            raise ValueError("native state initialization scale changed")
        self.bos_token_id = cfg.bos_token_id
        self.pad_token_id = cfg.pad_token_id
        if (type(self.bos_token_id) is not int or tokenizer.bos_token_id != self.bos_token_id
                or tokenizer.pad_token_id != self.pad_token_id):
            raise ValueError("tokenizer/config BOS or padding identity mismatch")
        self.layers = [LoopTap(block, 2 + 4 * recurrence + physical)
                       for recurrence in range(8) for physical, block in enumerate(stack.core_block)]
        self.layers += [LoopTap(block, -1 - physical) for physical, block in enumerate(stack.coda)]
        self.n_layers = len(self.layers)
        self.source_layers = tuple(range(32))
        self.target_layer = 33
        self.coda_layers = (32, 33)
        self.topology = [
            {"virtual": index, "stage": "core", "recurrence": index // 4,
             "physical": index % 4, "native_step_idx": 2 + index, "boundary": "native norm_4 output"}
            for index in range(32)
        ] + [{"virtual": 32 + physical, "stage": "coda", "physical": physical,
              "native_step_idx": -1 - physical, "boundary": "native block output before final norm"}
             for physical in range(2)]
        self.last_initialization = None

    @property
    def input_device(self):
        return self.hf_model.transformer.wte.weight.device

    @property
    def dtype(self):
        return self.hf_model.transformer.wte.weight.dtype

    def encode(self, text, *, max_length=128):
        """Configured BOS + plain text, no added EOS, no padding or chat template."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        _integer("max_length", max_length, 1)
        if max_length > self.hf_model.config.block_size:
            raise ValueError("maximum length exceeds the native positional table")
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=True,
                                 max_length=max_length, padding=False, return_attention_mask=False)
        ids = encoded.input_ids
        result = torch.tensor([ids], dtype=torch.long, device=self.input_device)
        self._validate_ids(result)
        return result

    def _validate_ids(self, input_ids):
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
                or input_ids.ndim != 2 or min(input_ids.shape) < 1
                or input_ids.device != self.input_device
                or input_ids.shape[1] > self.hf_model.config.block_size):
            raise ValueError("input_ids must be nonempty [B,S] int64 on the model device")
        if not torch.equal(input_ids, input_ids[:1].expand_as(input_ids)):
            raise ValueError("the adapter accepts one unpadded prompt replicated across lanes")
        tokens = input_ids[0].detach().cpu().tolist()
        if (tokens[0] != self.bos_token_id or self.pad_token_id in tokens
                or min(tokens) < 0 or max(tokens) >= self.hf_model.config.padded_vocab_size):
            raise ValueError("input tokens violate the configured BOS/unpadded/vocabulary contract")
        return tokens

    def _state_record(self, tokens):
        recipe = {"version": 1, "base_seed": self.state_seed,
                  "namespace": self.seed_namespace, "input_ids": tokens}
        digest = hashlib.sha256(_canonical(recipe)).digest()
        return {
            "base_seed": self.state_seed, "namespace": self.seed_namespace,
            "prompt_state_seed": int.from_bytes(digest[:8], "big") % (2 ** 63),
            "seed_recipe": "SHA256(canonical JSON(version, base_seed, namespace, input_ids)); first 8 bytes big-endian modulo 2^63",
            "input_ids_sha256": hashlib.sha256(_canonical(tokens)).hexdigest(),
            "native_draw": "randn([1,S,D]); overwrite trunc_normal(std=sqrt(2/(5D)), bounds=+-3std); multiply sqrt(D) in native dtype",
            "generator_device": str(self.input_device), "dtype": str(self.dtype),
            "one_prompt_shape": [1, len(tokens), self.d_model],
            "lane_coupling": "contiguous copies of the same one-prompt state",
            "test_time_noise": 0,
        }

    def initialization_metadata(self, input_ids):
        return self._state_record(self._validate_ids(input_ids))

    def _draw_state(self, input_ids, tokens):
        record = self._state_record(tokens)
        generator = torch.Generator(device=self.input_device)
        generator.manual_seed(record["prompt_state_seed"])
        # The otherwise discarded normal draw is part of native RNG ordering.
        state = torch.randn((1, len(tokens), self.d_model), dtype=self.dtype,
                            device=self.input_device, generator=generator)
        std = self.hf_model.config.init_values["std"]
        torch.nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        if self.hf_model.emb_scale != 1:
            state = state * self.hf_model.emb_scale
        self.last_initialization = record
        return state.expand(input_ids.shape[0], -1, -1).clone(memory_format=torch.contiguous_format)

    def sample_initial_state(self, input_ids):
        """Return coupled [B,S,D] states; retain its metadata for paired readouts."""
        return self._draw_state(input_ids, self._validate_ids(input_ids))

    def _check_state(self, state, input_ids, *, initial=False):
        if (not isinstance(state, torch.Tensor) or tuple(state.shape) != (*input_ids.shape, self.d_model)
                or state.dtype != self.dtype or state.device != self.input_device
                or state.layout != torch.strided or not torch.isfinite(state).all().item()):
            raise ValueError("state must be a finite complete [B,S,D] sequence in the native dtype/device")
        if initial and (state.requires_grad or not torch.equal(state, state[:1].expand_as(state))):
            raise ValueError("provided initial states must be fixed and identical across derivative lanes")

    def _prepare_state(self, input_ids, input_states):
        tokens = self._validate_ids(input_ids)
        if self.hf_model.config.test_time_noise != 0 or self.hf_model.gradient_checkpointing:
            raise ValueError("test-time noise and activation checkpointing must remain disabled")
        if input_states is None:
            return self._draw_state(input_ids, tokens)
        self._check_state(input_states, input_ids, initial=True)
        self.last_initialization = {
            "source": "explicit input_states; caller must retain the generating seed record",
            "one_prompt_state_sha256": hashlib.sha256(
                input_states[:1].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "lane_coupling": "validated identical initial states", "test_time_noise": 0,
        }
        return input_states.clone()

    def _coda_target(self, raw_core, freqs_cis):
        stack = self.hf_model.transformer
        hidden = stack.ln_f(raw_core)
        block_idx = torch.tensor(0, device="cpu", dtype=torch.long)
        for block in stack.coda:
            block_idx -= 1
            hidden = block(hidden, freqs_cis, block_idx, None, None)
        return hidden

    def forward(self, input_ids, *, input_states=None):
        """Native recurrence with gradients, returning pre-final-norm coda target."""
        hidden = self._prepare_state(input_ids, input_states)
        model = self.hf_model
        stack = model.transformer
        freqs_cis = model.freqs_cis[:, :input_ids.shape[1]]
        embedded = stack.wte(input_ids)
        if model.emb_scale != 1:
            embedded = embedded * model.emb_scale
        block_idx = torch.tensor(-1, device="cpu", dtype=torch.long)
        for block in stack.prelude:
            block_idx += 1
            embedded = block(embedded, freqs_cis, block_idx, None, None)
        for recurrence in range(8):
            hidden, block_idx = model.core_block_forward(
                hidden, embedded, freqs_cis, None, None, block_idx, recurrence,
            )
        return self._coda_target(hidden, freqs_cis)

    def unembed(self, residual):
        if not isinstance(residual, torch.Tensor) or residual.ndim < 1 or residual.shape[-1] != self.d_model:
            raise ValueError("residual must end in the native hidden width")
        hidden = residual.to(device=self.input_device, dtype=self.dtype)
        return self.hf_model.lm_head(self.hf_model.transformer.ln_f(hidden)).float()

    def coda_readout(self, raw_core, input_ids):
        """Trained readout on the complete raw sequence at positions 0..S-1.

        Apply position selection only to the returned logits. At the end of a
        core pass this is the native exit for that recurrence count; within a
        pass it is an explicitly interrupted-core diagnostic intervention.
        """
        self._validate_ids(input_ids)
        self._check_state(raw_core, input_ids)
        freqs_cis = self.hf_model.freqs_cis[:, :input_ids.shape[1]]
        return self.unembed(self._coda_target(raw_core, freqs_cis))

    def native_logits(self, input_ids, *, input_states=None):
        """Gradient-capable native (0,R) reference, using the same state recipe."""
        state = self._prepare_state(input_ids, input_states)
        return self.hf_model(
            input_ids=input_ids, input_states=state, num_steps=(0, 8), use_cache=False,
            output_details={"return_logits": True, "return_latents": False,
                            "return_head": False, "return_stats": False},
        ).logits


def load_huginn(snapshot, *, device="cpu", dtype=torch.bfloat16,
                state_seed=FIT_STATE_SEED, seed_namespace="calibration"):
    """Load local pinned Huginn inputs; no Hub lookup or download fallback.

    A production caller must validate the complete weight manifest first.
    ``verified_metadata_sha256`` describes the narrower checks done here.
    """
    snapshot = verify_pinned_metadata(snapshot)
    with (snapshot / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)
    for filename in set(index["weight_map"].values()):
        if Path(filename).name != filename or not (snapshot / filename).is_file():
            raise ValueError(f"missing or invalid local weight shard: {filename}")
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("unsupported native model dtype")
    RavenConfig, RavenForCausalLM = _native_classes(snapshot)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    config = RavenConfig.from_pretrained(str(snapshot), local_files_only=True)
    native = RavenForCausalLM.from_pretrained(str(snapshot), config=config, local_files_only=True,
                                             torch_dtype=dtype).to(device)
    result = HuginnLensModel(native, tokenizer, state_seed=state_seed, seed_namespace=seed_namespace)
    result.snapshot_path = snapshot
    result.declared_revision = REVISION
    result.verified_metadata_sha256 = dict(PINNED_SMALL_FILES)
    return result


@contextmanager
def _capture(adapter, run, *, gradients=False):
    """Self-test-only capture with guaranteed hook cleanup."""
    records = {}
    counts = [0] * adapter.n_layers
    handles = []
    def hook(index):
        def record(module, args, output):
            counts[index] += 1
            if gradients and index == 0:
                output.requires_grad_(True)
            records[index] = output
        return record
    try:
        for index, tap in enumerate(adapter.layers):
            handles.append(tap.register_forward_hook(hook(index)))
        output = run()
        if counts != [1] * 34:
            raise AssertionError(f"virtual hook call counts differ: {counts}")
        yield output, records
    finally:
        for handle in handles:
            handle.remove()


def self_test(snapshot):
    """Exercise native tiny CPU models; never deserialize checkpoint weights."""
    from transformers import AutoTokenizer
    snapshot = verify_pinned_metadata(snapshot)
    RavenConfig, RavenForCausalLM = _native_classes(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    for text in ("The river is wide.", "  Arithmetic: 7 + 8", "Zašto nebo?", ""):
        for maximum in (1, 4, 16):
            actual = tokenizer(text, add_special_tokens=True, truncation=True, max_length=maximum,
                               padding=False).input_ids
            expected = [65504, *tokenizer(text, add_special_tokens=False).input_ids[:maximum - 1]]
            if actual != expected:
                raise AssertionError("native configured BOS/token truncation mismatch")

    class TinyTokenizer:
        bos_token_id, eos_token_id, pad_token_id = 1, 2, 31
        def __call__(self, text, *, add_special_tokens=True, max_length=128, **kwargs):
            ids = [3 + ord(character) % 25 for character in text]
            if add_special_tokens:
                ids = [self.bos_token_id, *ids]
            return types.SimpleNamespace(input_ids=ids[:max_length])

    torch.set_num_threads(1)
    cases = []
    fp32_adapter = None
    for dtype in (torch.float32, torch.bfloat16):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260908)
            config = RavenConfig(n_embd=8, n_heads=2, intermediate_size=16, vocab_size=32,
                                 block_size=16, bos_token_id=1, eos_token_id=2, pad_token_id=31)
            native = RavenForCausalLM(config)
            native.to(dtype=dtype)
            # Native rope construction is explicitly FP32 and the checkpoint
            # stores that buffer in FP32, independent of parameter dtype.
            native.freqs_cis = native._precompute_freqs_cis()
            with torch.no_grad():
                native.transformer.ln_f.weight.copy_(torch.linspace(0.65, 1.35, 8, dtype=dtype))
        adapter = HuginnLensModel(native, TinyTokenizer(), cpu_test=True)
        ids = adapter.encode("abcd", max_length=5)
        if dtype == torch.float32:
            fp32_adapter = adapter
        one = adapter.sample_initial_state(ids)
        seed = adapter.initialization_metadata(ids)["prompt_state_seed"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            native_init = native.initialize_state(torch.empty_like(one))
        if not torch.equal(one.view(torch.uint8), native_init.view(torch.uint8)):
            raise AssertionError("private generator draw differs from native B=1 initialization")
        saved_rng = torch.get_rng_state().clone()
        for batch in (1, 2, 3, 8):
            replicated = ids.repeat(batch, 1)
            initial = adapter.sample_initial_state(replicated)
            if not torch.equal(initial, one.expand_as(initial)) or not torch.equal(saved_rng, torch.get_rng_state()):
                raise AssertionError("replicas or global RNG isolation changed")
            with _capture(adapter, lambda: adapter.forward(replicated, input_states=initial)) as (target, records):
                adapter_logits = adapter.unembed(target)
                raw = records[31]
                if not torch.equal(target, records[33]):
                    raise AssertionError("forward target is not the last coda block output")
            with _capture(adapter, lambda: adapter.native_logits(replicated, input_states=initial)) as (native_logits, native_records):
                if not torch.equal(adapter_logits, native_logits):
                    raise AssertionError("native (0,R) and adapter logits differ")
                for cell in range(34):
                    if not torch.equal(records[cell], native_records[cell]):
                        raise AssertionError(f"native and adapter virtual states differ at {cell}")
            coda = adapter.coda_readout(raw, replicated)
            predicted = native.predict_from_latents(raw).logits
            if not torch.equal(coda, native_logits) or not torch.equal(coda, predicted):
                raise AssertionError("raw-core trained coda boundary differs from native readout")
            if torch.equal(adapter.unembed(raw), coda):
                raise AssertionError("test weights fail to distinguish raw and trained coda readouts")
            replica_bitwise = all(torch.equal(records[cell], records[cell][:1].expand_as(records[cell]))
                                  for cell in range(34))
            replica_max_abs = max((records[cell].float() - records[cell][:1].float()).abs().max().item()
                                  for cell in range(34))
            # Identical input states are the coupling contract. Native kernels
            # can still introduce lane-dependent rounding, already checked above
            # against the same-shaped native forward at every virtual event.
            cases.append({"dtype": str(dtype), "batch": batch, "native_state_and_logit_parity": "bitwise",
                          "virtual_hook_counts": [1] * 34, "coupled_initial_lanes": "bitwise",
                          "native_intermediate_lanes_bitwise_equal": replica_bitwise,
                          "native_intermediate_lanes_max_absolute": replica_max_abs})
        alternative = HuginnLensModel(native, TinyTokenizer(), seed_namespace="evaluation", cpu_test=True)
        if torch.equal(alternative.sample_initial_state(ids), one):
            raise AssertionError("evaluation namespace did not separate the initial state")
        try:
            adapter.coda_readout(raw[:, -1], replicated)
        except ValueError:
            pass
        else:
            raise AssertionError("coda accepted a token-vector-only state")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in native.parameters()):
            raise AssertionError("native parameter gradients were enabled")

    adapter = fp32_adapter
    ids = adapter.encode("abcd", max_length=5)
    initial = adapter.sample_initial_state(ids)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8262)
        cotangent = torch.randn((1, 5, 8))
        cotangent /= cotangent.norm()
    with _capture(adapter, lambda: adapter.forward(ids, input_states=initial), gradients=True) as (target, records):
        gradients = torch.autograd.grad(target, [records[cell] for cell in range(32)], cotangent)
    with _capture(adapter, lambda: adapter.native_logits(ids, input_states=initial), gradients=True) as (_, native_records):
        native_gradients = torch.autograd.grad(native_records[33], [native_records[cell] for cell in range(32)], cotangent)
    for cell, (gradient, expected) in enumerate(zip(gradients, native_gradients)):
        if not torch.isfinite(gradient).all() or gradient.norm() == 0 or not torch.equal(gradient, expected):
            raise AssertionError(f"source {cell} has disconnected/nonfinite or non-native derivatives")

    differences = []
    for cell in (0, 4, 31):
        direction = gradients[cell] / gradients[cell].norm()
        analytic = (gradients[cell] * direction).sum().item()
        numerical = []
        passing = []
        for epsilon in (1e-2, 3e-3, 1e-3):
            values = []
            for sign in (1, -1):
                handle = adapter.layers[cell].register_forward_hook(
                    lambda module, args, output, sign=sign, epsilon=epsilon: output + sign * epsilon * direction
                )
                try:
                    with torch.no_grad():
                        values.append((adapter.forward(ids, input_states=initial) * cotangent).sum().item())
                finally:
                    handle.remove()
            estimate = (values[0] - values[1]) / (2 * epsilon)
            numerical.append({"epsilon": epsilon, "derivative": estimate})
            passing.append(abs(estimate - analytic) <= 2e-5 + 0.02 * abs(analytic))
        if not any(a and b for a, b in zip(passing, passing[1:])):
            raise AssertionError(f"directional derivative failed at source {cell}: {analytic}, {numerical}")
        differences.append({"source": cell, "analytic": analytic, "numerical": numerical,
                            "adjacent_passing_steps": True})

    with _capture(adapter, lambda: adapter.forward(ids, input_states=initial)) as (_, records):
        raw = records[31].clone()
    original = adapter.coda_readout(raw, ids)
    changed = raw.clone()
    changed[:, :-1] = changed[:, :-1].flip(-1)
    if torch.equal(original[:, -1], adapter.coda_readout(changed, ids)[:, -1]):
        raise AssertionError("coda test failed to exercise attention to earlier sequence states")
    before = [len(module._forward_hooks) for module in adapter.hf_model.modules()]
    try:
        with _capture(adapter, lambda: adapter.forward(ids, input_states=initial)):
            raise RuntimeError("deliberate hook-cleanup check")
    except RuntimeError:
        pass
    after = [len(module._forward_hooks) for module in adapter.hf_model.modules()]
    if before != after or any(parameter.grad is not None for parameter in adapter.hf_model.parameters()):
        raise AssertionError("hooks or parameter gradients leaked")
    def deliberate_failure(module, args, output):
        raise RuntimeError("deliberate forward failure")
    failure_handle = adapter.layers[7].register_forward_hook(deliberate_failure)
    try:
        try:
            with _capture(adapter, lambda: adapter.forward(ids, input_states=initial)):
                raise AssertionError("forward failure was not triggered")
        except RuntimeError as error:
            if str(error) != "deliberate forward failure":
                raise
    finally:
        failure_handle.remove()
    if before != [len(module._forward_hooks) for module in adapter.hf_model.modules()]:
        raise AssertionError("forward exception leaked hooks")
    guards = []
    def rejects(label, function):
        try:
            function()
        except ValueError:
            guards.append(label)
        else:
            raise AssertionError(f"guard did not reject: {label}")
    mismatched_ids = ids.repeat(2, 1)
    mismatched_ids[1, -1] = 6
    rejects("different prompts across lanes", lambda: adapter.forward(mismatched_ids))
    paired_ids = ids.repeat(2, 1)
    mismatched_states = initial.repeat(2, 1, 1)
    mismatched_states[1, 0, 0] += 1
    rejects("different initial states across lanes",
            lambda: adapter.forward(paired_ids, input_states=mismatched_states))
    padded = ids.clone()
    padded[0, -1] = adapter.pad_token_id
    rejects("padding", lambda: adapter.forward(padded))
    rejects("shortened coda sequence", lambda: adapter.coda_readout(raw[:, -1:], ids))
    adapter.hf_model.config.test_time_noise = 0.1
    try:
        rejects("test-time noise", lambda: adapter.forward(ids))
    finally:
        adapter.hf_model.config.test_time_noise = 0
    adapter.hf_model.gradient_checkpointing = True
    try:
        rejects("activation checkpointing", lambda: adapter.forward(ids))
    finally:
        adapter.hf_model.gradient_checkpointing = False
    if torch.cuda.is_initialized():
        raise AssertionError("CPU self-test initialized CUDA")
    return {
        "status": "passed", "scope": "tiny native Raven CPU instances; no Huginn checkpoint weights loaded",
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_metadata_sha256": PINNED_SMALL_FILES, "torch_version": torch.__version__,
        "configured_bos_parity": True, "initial_rng_order_parity": "bitwise at native B=1",
        "global_rng_preserved": True, "namespace_separation": True, "cases": cases,
        "all_32_source_vjps_match_native": "bitwise", "directional_derivatives": differences,
        "finite_difference_tolerance": {"absolute": 2e-5, "relative": 0.02,
                                        "adjacent_passing_steps_required": 2},
        "coda_uses_full_sequence_context": True, "hook_cleanup": True,
        "forward_exception_hook_cleanup": True, "input_and_runtime_guards": guards,
        "parameters_frozen": True, "cuda_initialized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(self_test(args.snapshot), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
