"""Build a sealed local source bundle; never upload, rent or run a model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


REPO = Path(__file__).resolve().parents[3]
ROUND = REPO / "research/refit_round_2026-09-07"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(ouro_src: Path, destination: Path) -> dict:
    if destination.exists() or destination.with_suffix(".manifest.json").exists():
        raise FileExistsError("bundle or manifest already exists; choose a new name")
    files: dict[str, bytes] = {}

    def add(path: Path, archive_name: str) -> None:
        if archive_name in files:
            raise ValueError(f"duplicate archive member: {archive_name}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source is missing or linked: {path}")
        files[archive_name] = path.read_bytes()

    for path in sorted((REPO / "jlens").rglob("*.py")):
        add(path, "jacobian-lens/" + path.relative_to(REPO).as_posix())
    for name in ("LICENSE", "README.md", "pyproject.toml", "tests/__init__.py", "tests/tiny.py"):
        add(REPO / name, "jacobian-lens/" + name)
    for name in ("__init__.py", "recurrent.py", "bench.py", "evidence.py", "fit_lens.py", "evaluate.py", "evaldata.py"):
        add(ouro_src / "ouro_jlens" / name, "ouro_project/src/ouro_jlens/" + name)
    names = (
        "fit_estimators.py",
        "audit/calibration_plan.json", "audit/calibration_verification.json", "audit/EVIDENCE.md",
        "design_audit/CONTRACT.md", "design_audit/verify_estimators.py",
        "optimization/optimized_fitting.py", "optimization/cuda_graph_candidate.py",
        "optimization/saved_tensor_candidate.py", "optimization/benchmark_optimized.py",
        "optimization/bench_offload_reference.py",
        "optimization/RESULTS.md", "optimization/FINAL_VERIFICATION.json",
        "optimization/integrated_final_cpu.json", "optimization/CONTRACT.md",
        "GPU_AND_BUDGET_AFTER_OPTIMIZATION.md", "HUGINN_BUDGET.md",
        "deployment/README.md", "deployment/environment.json", "deployment/model_manifest.json",
        "deployment/run_spec.json", "deployment/run_refits.py", "deployment/prepare_bundle.py",
        "deployment/ENVIRONMENT.md", "deployment/environment_portability.json",
        "deployment/requirements.txt", "deployment/requirements.lock",
        "deployment/wheels.json", "deployment/wheel_candidates.json",
        "deployment/cli_checks_v1.json", "deployment/runner_cpu_v3.json",
        "deployment/combined_contract.json", "deployment/control_estimators.py", "deployment/run_controls.py",
        "deployment/evaluate_refits.py", "deployment/evaluate_controls.py", "deployment/evaluate_huginn.py",
        "deployment/huginn_adapter.py", "deployment/run_huginn.py", "deployment/huginn_model_manifest.json",
        "deployment/huginn_calibration.json", "deployment/huginn_eligibility.json", "deployment/preflight_gpu.py",
        "deployment/cloud_worker.py", "deployment/pod_entry.sh", "deployment/fresh_environment_v1.json",
        "deployment/control_cpu_v1.json", "deployment/controls_review_v1.json", "deployment/huginn_runner_cpu_v1.json",
        "deployment/huginn_evaluator_cpu_v1.json", "deployment/huginn_owner_cpu_v1.json",
        "deployment/huginn_output_cpu_v1.json", "deployment/controls_evaluator_cpu_v1.json",
        "deployment/controls_historical_parity_v1.json", "deployment/main_evaluator_cpu_v1.json",
        "deployment/huginn_adapter_cpu_v1.json", "deployment/fresh_runner_cpu_v1.json",
        "deployment/fresh_controls_cpu_v1.json", "deployment/fresh_huginn_adapter_cpu_v1.json",
        "deployment/fresh_evaluator_cpu_v1.json",
    )
    for name in names:
        path = ROUND / name
        add(path, "jacobian-lens/" + path.relative_to(REPO).as_posix())
    # These controllers are preserved for review; only cloud_worker runs on
    # the GPU. No key file or account record is part of this source bundle.
    for name in ("lease.py", "runpod_api.py", "stage_pod.py"):
        path = ROUND / "deployment" / name
        add(path, "jacobian-lens/" + path.relative_to(REPO).as_posix())
    for path in sorted((ROUND / "deployment").glob("*_cpu_v1.json")):
        name = "jacobian-lens/" + path.relative_to(REPO).as_posix()
        if name not in files:
            add(path, name)

    spec = json.loads((ROUND / "deployment/run_spec.json").read_bytes())
    all_texts = []
    for fit in spec["fits"]:
        path = (ROUND / "deployment" / fit["prompts_path"]).resolve()
        path.relative_to(ROUND / "audit")
        data = path.read_bytes()
        prompts = json.loads(data)
        if digest(data) != fit["sha256"] or len(prompts) != fit["n_prompts"]:
            raise ValueError(f"calibration identity/count mismatch: {path}")
        if fit["n_prompts"] != 100 or not all(isinstance(item, str) for item in prompts):
            raise ValueError("calibration must contain 100 strings")
        all_texts.extend(prompts)
        add(path, "jacobian-lens/" + path.relative_to(REPO).as_posix())
    if len(spec["fits"]) != 5 or len(all_texts) != 500 or len(set(all_texts)) != 500:
        raise ValueError("expected five disjoint N100 calibration lists")

    for source in spec["source_files"]:
        prefix = {"jlens": "jacobian-lens/", "ouro_src": "ouro_project/src/"}[source["root"]]
        name = prefix + source["path"]
        if name not in files:
            root = {"jlens": REPO, "ouro_src": ouro_src}[source["root"]]
            relative = Path(source["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid source path in run specification")
            add(root / relative, name)
        if digest(files[name]) != source["sha256"]:
            raise ValueError(f"run spec source differs from bundle: {name}")
    for entry in (spec["model"], spec["environment"]):
        name = entry.get("manifest_path", entry.get("path"))
        expected = entry.get("manifest_sha256", entry.get("sha256"))
        data = (ROUND / "deployment" / name).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"run spec resource digest mismatch: {name}")

    manifest = {
        "schema_version": 1,
        "purpose": "Local source/calibration staging; no model execution or external transfer.",
        "run_spec_sha256": digest((ROUND / "deployment/run_spec.json").read_bytes()),
        "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(files.items())},
        "calibration": {"fits": 5, "prompts_each": 100, "total_distinct_texts": 500},
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    members = {**files, "MANIFEST.json": manifest_bytes}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, "r:gz") as archive:
        if sorted(archive.getnames()) != sorted(members):
            raise ValueError("archive member list mismatch")
        for member in archive:
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"unexpected archive entry: {member.name}")
            if archive.extractfile(member).read() != members[member.name]:
                raise ValueError(f"archive bytes differ: {member.name}")
    with destination.with_suffix(".manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    return {
        "status": "passed", "archive": destination.name,
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": digest(destination.read_bytes()),
        "source_files": len(files), "archive_members": len(members),
        "calibration_counts": [100] * 5, "distinct_calibration_texts": 500,
        "all_archive_members_verified": True,
        "gpu_operations": 0, "external_transfers": 0, "cloud_operations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ouro-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.ouro_src, args.output), indent=2))


if __name__ == "__main__":
    main()
