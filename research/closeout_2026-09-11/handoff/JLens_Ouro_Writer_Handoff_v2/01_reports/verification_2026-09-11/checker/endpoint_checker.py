#!/usr/bin/env python3
"""Independent endpoint checker for the Ouro J-Lens prospective confirmation (160 items).

Written only from the delegated specification, PROSPECTIVE_PLAN.md and the data files.
No producer or replay code was imported, copied or opened. Standard library + numpy only.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

ARMS = ("raw", "fit01", "fit02", "penultimate", "sampled_sum", "diagonal")
N_COLUMNS = {"raw": 192, "fit01": 192, "fit02": 192, "penultimate": 190, "sampled_sum": 191, "diagonal": 191}
N_ITEMS = 160
N_NAMES = 80
NAME_ROWS = 128
VOCAB = 49152
TOPK = 10
BAND = tuple(range(169, 181))
N_REPLICATES = 20000
CHUNK = 500
SEEDS = {"family": 2026090901, "item": 2026090902, "secondary": 2026090903}
VALUE_ORDER = ("primary", "fit01_intended_B", "fit01_control_B", "raw_intended_B", "raw_control_B",
               "intended_difference", "control_difference")

CHECKER_PATH = Path(__file__).resolve()
FIXTURE_DIR = CHECKER_PATH.parent / "fixtures"
SOURCE_RESULTS = Path("/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09/cloud_leases/attempt_06/handoff/"
                      "accepted/ouro_confirmation_20260911_fixed160_native1/"
                      "ff7c0769b2b0dbe2e53b120daf7f0729434fe65e54c013288771f5052ffd4cef/results")
SOURCE_ANALYSIS = Path("/home/moloch/ouro_project/jacobian-lens/research/confirmation_2026-09-09/results/"
                       "ouro_confirmation_20260911_fixed160_native1/analysis/analysis.json")


# ----------------------------------------------------------------------------- utilities

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a):
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(repr(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def write_json_exclusive(path, doc):
    with open(path, "x") as fh:  # never overwrite
        json.dump(doc, fh, indent=1, allow_nan=False)
        fh.write("\n")


class Item:
    __slots__ = ("position", "slots", "group")

    def __init__(self, position, slots, group):
        self.position = position   # row index in population / readout axis 0
        self.slots = slots         # list of (own_index, np.array(control_indices)) for eligible slots
        self.group = group


# ----------------------------------------------------------------------------- input checks

def validate_population(pop):
    """Return (failures, notes, items)."""
    failures, notes, items = [], [], []
    try:
        names = pop["names"]["multihop"]
    except Exception as exc:  # noqa: BLE001
        return [f"population: cannot read names.multihop ({exc!r})"], notes, None
    if not isinstance(names, list) or len(names) != N_NAMES:
        failures.append(f"population: names.multihop must be a list of {N_NAMES} names")
    rows = pop.get("rows") if isinstance(pop, dict) else None
    if not isinstance(rows, list) or len(rows) != N_ITEMS:
        n = len(rows) if isinstance(rows, list) else None
        failures.append(f"population: rows must be a list of {N_ITEMS}, got length {n}")
        return failures, notes, None
    own_padded_rows = []
    for i, row in enumerate(rows):
        try:
            eligible, own, ctrl, group = (row["eligible"], row["own_index"], row["control_indices"],
                                          row["dependency_group_id"])
        except (KeyError, TypeError) as exc:
            failures.append(f"row {i}: missing field {exc!r}")
            continue
        if not (isinstance(eligible, list) and all(isinstance(e, bool) for e in eligible)):
            failures.append(f"row {i}: eligible must be a list of booleans")
            continue
        if not (isinstance(own, list) and all(is_int(o) for o in own)):
            failures.append(f"row {i}: own_index must be a list of ints")
            continue
        if not (isinstance(ctrl, list) and all(isinstance(c, list) and all(is_int(k) for k in c) for c in ctrl)):
            failures.append(f"row {i}: control_indices must be a list of int lists")
            continue
        if not isinstance(group, str):
            failures.append(f"row {i}: dependency_group_id must be a string")
            continue
        n_slots = len(eligible)
        if len(ctrl) != n_slots:
            failures.append(f"row {i}: {len(ctrl)} control_indices lists for {n_slots} eligible slots")
            continue
        if len(own) != n_slots:
            # Data/spec disagreement: own_index is longer than eligible, padded with -1.
            if len(own) > n_slots and all(o == -1 for o in own[n_slots:]):
                own_padded_rows.append(i)
            else:
                failures.append(f"row {i}: own_index length {len(own)} vs eligible length {n_slots}")
                continue
        expected_ctrl = [k for k in range(N_NAMES) if k not in own]
        slots, row_ok = [], True
        for s in range(n_slots):
            if not eligible[s]:
                continue
            if not 0 <= own[s] < N_NAMES:
                failures.append(f"row {i} slot {s}: own_index {own[s]} outside [0, {N_NAMES})")
                row_ok = False
                continue
            if ctrl[s] != expected_ctrl:
                failures.append(f"row {i} slot {s}: control_indices ({len(ctrl[s])} entries) != all names "
                                f"except own_index ({len(expected_ctrl)} entries)")
                row_ok = False
                continue
            if not expected_ctrl:
                failures.append(f"row {i} slot {s}: empty control set")
                row_ok = False
                continue
            slots.append((own[s], np.array(expected_ctrl, dtype=np.int64)))
        if row_ok and slots:
            items.append(Item(i, slots, group))
    if own_padded_rows:
        notes.append({
            "specification_disagreement": "own_index is not one entry per label slot",
            "detail": "len(own_index) > len(eligible) == len(control_indices); the extra own_index entries are all -1",
            "rows_affected": len(own_padded_rows),
            "handling": "slots = range(len(eligible)); trailing own_index entries must be exactly -1 or the input is "
                        "rejected; control check uses the whole own_index list as specified",
        })
    if not items and not failures:
        failures.append("population: no eligible items")
    return failures, notes, items


def validate_readouts(arrays):
    failures = []
    for arm in ARMS:
        a = arrays.get(arm)
        if a is None:
            failures.append(f"{arm}: allrank missing")
            continue
        if a.dtype != np.int32:
            failures.append(f"{arm}: allrank dtype {a.dtype}, expected int32")
        shape = (N_ITEMS, NAME_ROWS, N_COLUMNS[arm])
        if a.shape != shape:
            failures.append(f"{arm}: allrank shape {a.shape}, expected {shape}")
            continue
        real = a[:, :N_NAMES, :]
        bad = np.argwhere((real < 0) | (real >= VOCAB))
        if len(bad):
            i, j, c = (int(x) for x in bad[0])
            failures.append(f"{arm}: {len(bad)} name ranks outside [0, {VOCAB}); first at item {i} name {j} "
                            f"column {c} value {int(a[i, j, c])}")
        pad = a[:, N_NAMES:, :]
        badpad = np.argwhere(pad != -1)
        if len(badpad):
            i, j, c = (int(x) for x in badpad[0])
            failures.append(f"{arm}: {len(badpad)} padding entries != -1; first at item {i} row {j + N_NAMES} "
                            f"column {c} value {int(a[i, j + N_NAMES, c])}")
    return failures


# ----------------------------------------------------------------------------- statistics

def arm_curves(allrank, items):
    hit = (allrank >= 0) & (allrank < TOPK)
    n, cols = len(items), allrank.shape[2]
    intended, control = np.empty((n, cols)), np.empty((n, cols))
    for r, it in enumerate(items):
        h = hit[it.position].astype(np.float64)
        intended[r] = np.mean([h[o] for o, _ in it.slots], axis=0)
        control[r] = np.mean([h[ctrl].mean(axis=0) for _, ctrl in it.slots], axis=0)
    return hit, intended, control


def any_layer_item(hit, items, region):
    anyhit = hit[:, :, region].any(axis=2).astype(np.float64)
    out = np.empty(len(items))
    for r, it in enumerate(items):
        a = anyhit[it.position]
        out[r] = np.mean([a[o] - a[ctrl].mean() for o, ctrl in it.slots])
    return out


def fixed_mean(values, region):
    return values[:, list(region)].mean(axis=1)


def group_bootstrap(X, labels, seed):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    uniq = list(dict.fromkeys(labels))
    index = {u: g for g, u in enumerate(uniq)}
    g_of = np.array([index[l] for l in labels])
    G = len(uniq)
    sums = np.stack([X[g_of == g].sum(axis=0) for g in range(G)])
    sizes = np.bincount(g_of, minlength=G)
    rng = np.random.default_rng(seed)
    reps = []
    for start in range(0, N_REPLICATES, CHUNK):
        draw = rng.integers(G, size=(min(CHUNK, N_REPLICATES - start), G))
        reps.append(sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None])
    return np.concatenate(reps, axis=0)


def secondary_specs(band):
    b = list(band)

    def spec(cid, left, right, region, metric="fixed_mean"):
        return {"id": cid, "left": left, "right": right, "region": list(region), "metric": metric}

    specs = []
    for L in (1, 2, 3):
        for metric in ("fixed_mean", "any_layer"):
            specs.append(spec(f"early_loop{L}_{metric}", "fit01", "raw", range(48 * (L - 1), 48 * L), metric))
    specs.append(spec("fit01_final_third", "fit01", "raw", range(176, 192)))
    specs.append(spec("fit01_layer32", "fit01", "raw", [175]))
    specs.append(spec("fit02_local", "fit02", "raw", b))
    specs.append(spec("fit01_minus_fit02_local", "fit01", "fit02", b))
    for arm in ("penultimate", "sampled_sum", "diagonal"):
        specs.append(spec(f"{arm}_local", arm, "raw", b))
        specs.append(spec(f"{arm}_final_third", arm, "raw", range(176, 190 if arm == "penultimate" else 191)))
    specs.append(spec("fit01_minus_penultimate_local", "fit01", "penultimate", b))
    specs.append(spec("fit01_minus_penultimate_final_third", "fit01", "penultimate", range(176, 190)))
    specs.append(spec("sampled_sum_minus_diagonal_local", "sampled_sum", "diagonal", b))
    specs.append(spec("sampled_sum_minus_diagonal_final_third", "sampled_sum", "diagonal", range(176, 191)))
    return specs


def rebuild(items, arrays, band=BAND, seeds=None):
    seeds = dict(SEEDS if seeds is None else seeds)
    band = list(band)
    hits, curves = {}, {}
    for arm in ARMS:
        hit, intended, control = arm_curves(arrays[arm], items)
        hits[arm] = hit
        curves[arm] = {"own": intended, "control": control, "excess": intended - control}
    f, r = curves["fit01"], curves["raw"]
    v2, v3 = fixed_mean(f["own"], band), fixed_mean(f["control"], band)
    v4, v5 = fixed_mean(r["own"], band), fixed_mean(r["control"], band)
    v1 = fixed_mean(f["excess"], band) - fixed_mean(r["excess"], band)
    X = np.column_stack([v1, v2, v3, v4, v5, v2 - v4, v3 - v5])
    labels = [it.group for it in items]

    fam = np.quantile(group_bootstrap(X, labels, seeds["family"]), [0.025, 0.975], axis=0)
    item_iv = np.quantile(group_bootstrap(X[:, 0], [str(it.position) for it in items], seeds["item"])[:, 0],
                          [0.025, 0.975])

    heterogeneity = []
    lab = np.array(labels, dtype=object)
    for g in dict.fromkeys(labels):
        m = lab == g
        heterogeneity.append({
            "group": g, "items": int(m.sum()), "primary_mean": float(X[m, 0].mean()),
            "without_group": float(X[~m, 0].mean()) if (~m).any() else None,
            "intended_difference": float(X[m, 5].mean()), "control_difference": float(X[m, 6].mean()),
        })
    without = [h["without_group"] for h in heterogeneity if h["without_group"] is not None]

    specs = secondary_specs(band)
    columns = []
    for sp in specs:
        for side in (sp["left"], sp["right"]):
            if max(sp["region"]) >= N_COLUMNS[side]:
                raise ValueError(f"{sp['id']}: region exceeds {side} columns")
        if sp["metric"] == "fixed_mean":
            columns.append(fixed_mean(curves[sp["left"]]["excess"], sp["region"])
                           - fixed_mean(curves[sp["right"]]["excess"], sp["region"]))
        else:
            columns.append(any_layer_item(hits[sp["left"]], items, sp["region"])
                           - any_layer_item(hits[sp["right"]], items, sp["region"]))
    Xs = np.column_stack(columns)
    Ys = group_bootstrap(Xs, labels, seeds["secondary"])
    e = Xs.mean(axis=0)
    sd_raw = Ys.std(axis=0, ddof=1)
    sd = np.where(sd_raw == 0, 1.0, sd_raw)
    q = float(np.quantile(np.max(np.abs((Ys - e) / sd), axis=1), 0.95))
    secondary = [dict(sp, estimate=float(e[k]), simultaneous_95_interval=[float(e[k] - q * sd[k]), float(e[k] + q * sd[k])])
                 for k, sp in enumerate(specs)]

    rebuilt = {
        "eligible_items": len(items),
        "groups": len(dict.fromkeys(labels)),
        "estimates": X.mean(axis=0).tolist(),
        "family_percentile_95_intervals": fam.T.tolist(),
        "primary_item_resampling_95_interval": item_iv.tolist(),
        "group_heterogeneity": heterogeneity,
        "leave_one_group_range": [min(without), max(without)] if without else None,
        "secondary_family": secondary,
        "secondary_max_t_95_quantile": q,
        "descriptive_curves": {arm: {k: curves[arm][k].mean(axis=0).tolist() for k in ("own", "control", "excess")}
                               for arm in ARMS},
        "value_order": list(VALUE_ORDER),
        "item_positions": [it.position for it in items],
        "item_values": X.tolist(),
        "secondary_item_values": Xs.tolist(),
        "secondary_zero_sd_ids": [specs[k]["id"] for k in np.flatnonzero(sd_raw == 0)],
    }
    internal = {"X": X, "Xs": Xs, "curves": curves, "items": items}
    return rebuilt, internal


# ----------------------------------------------------------------------------- comparison

def _walk_numeric(a, b, path, diffs, problems):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} != expected {len(b)}")
            return
        for k, (x, y) in enumerate(zip(a, b)):
            _walk_numeric(x, y, f"{path}[{k}]", diffs, problems)
    elif is_number(a) and is_number(b):
        if not (math.isfinite(a) and math.isfinite(b)):
            problems.append(f"{path}: non-finite {a!r} vs {b!r}")
        else:
            diffs.append((abs(float(a) - float(b)), path))
    else:
        problems.append(f"{path}: type {type(a).__name__} vs expected {type(b).__name__}")


def _walk_exact(a, b, path, problems):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} != expected {len(b)}")
            return
        for k, (x, y) in enumerate(zip(a, b)):
            _walk_exact(x, y, f"{path}[{k}]", problems)
    elif type(a) is not type(b) or a != b:
        problems.append(f"{path}: {a!r} != expected {b!r}")


def _record(name, tol, diffs=None, problems=(), kind="numeric"):
    diffs = diffs or []
    mx = max(diffs) if diffs else None
    passed = not problems and (mx is None or mx[0] <= tol)
    return {"quantity": name, "kind": kind, "passed": passed, "n_values": len(diffs),
            "max_abs_diff": mx[0] if mx else None, "max_abs_diff_at": mx[1] if mx else None,
            "n_problems": len(problems), "problems": list(problems)[:20]}


def compare(rebuilt, expected, tol):
    out = []

    def get(key):
        return expected[key] if isinstance(expected, dict) and key in expected else KeyError

    def numeric(key, a=None, b=None, name=None):
        a = rebuilt[key] if a is None else a
        b = get(key) if b is None else b
        if b is KeyError:
            out.append(_record(name or key, tol, problems=[f"{key}: missing in expected"]))
            return
        diffs, problems = [], []
        _walk_numeric(a, b, name or key, diffs, problems)
        out.append(_record(name or key, tol, diffs, problems))

    def exact(key, a=None, b=None, name=None):
        a = rebuilt[key] if a is None else a
        b = get(key) if b is None else b
        problems = [f"{key}: missing in expected"] if b is KeyError else []
        if not problems:
            _walk_exact(a, b, name or key, problems)
        out.append(_record(name or key, tol, problems=problems, kind="exact"))

    exact("eligible_items")
    exact("groups")
    numeric("estimates")
    numeric("family_percentile_95_intervals")
    numeric("primary_item_resampling_95_interval")

    het_keys = ["group", "items", "primary_mean", "without_group", "intended_difference", "control_difference"]
    eh = get("group_heterogeneity")
    if not isinstance(eh, list) or len(eh) != len(rebuilt["group_heterogeneity"]):
        msg = f"group_heterogeneity: expected list of {len(rebuilt['group_heterogeneity'])}"
        out.append(_record("group_heterogeneity.identity", tol, problems=[msg], kind="exact"))
        out.append(_record("group_heterogeneity.values", tol, problems=[msg]))
    else:
        probs, diffs, nprobs = [], [], []
        for k, (x, y) in enumerate(zip(rebuilt["group_heterogeneity"], eh)):
            p = f"group_heterogeneity[{k}]"
            if not isinstance(y, dict) or sorted(y) != sorted(het_keys):
                probs.append(f"{p}: keys {sorted(y) if isinstance(y, dict) else y!r} != {sorted(het_keys)}")
                continue
            _walk_exact([x["group"], x["items"]], [y["group"], y["items"]], f"{p}.[group,items]", probs)
            for key in het_keys[2:]:
                _walk_numeric(x[key], y[key], f"{p}.{key}", diffs, nprobs)
        out.append(_record("group_heterogeneity.identity", tol, problems=probs, kind="exact"))
        out.append(_record("group_heterogeneity.values", tol, diffs, nprobs))
    numeric("leave_one_group_range")

    sec_keys = ["estimate", "id", "left", "metric", "region", "right", "simultaneous_95_interval"]
    es = get("secondary_family")
    names = ("secondary_family.identity", "secondary_family.estimate", "secondary_family.simultaneous_95_interval")
    if not isinstance(es, list) or len(es) != len(rebuilt["secondary_family"]):
        msg = f"secondary_family: expected list of {len(rebuilt['secondary_family'])}"
        for n in names:
            out.append(_record(n, tol, problems=[msg], kind="exact" if n.endswith("identity") else "numeric"))
    else:
        probs, d_est, p_est, d_iv, p_iv = [], [], [], [], []
        for k, (x, y) in enumerate(zip(rebuilt["secondary_family"], es)):
            p = f"secondary_family[{k}]"
            if not isinstance(y, dict) or sorted(y) != sec_keys:
                probs.append(f"{p}: keys {sorted(y) if isinstance(y, dict) else y!r} != {sec_keys}")
                continue
            for key in ("id", "left", "right", "region", "metric"):
                _walk_exact(x[key], y[key], f"{p}.{key}", probs)
            _walk_numeric(x["estimate"], y["estimate"], f"{p}({x['id']}).estimate", d_est, p_est)
            _walk_numeric(x["simultaneous_95_interval"], y["simultaneous_95_interval"],
                          f"{p}({x['id']}).simultaneous_95_interval", d_iv, p_iv)
        out.append(_record(names[0], tol, problems=probs, kind="exact"))
        out.append(_record(names[1], tol, d_est, p_est))
        out.append(_record(names[2], tol, d_iv, p_iv))
    numeric("secondary_max_t_95_quantile")

    ec = get("descriptive_curves")
    if not isinstance(ec, dict) or sorted(ec) != sorted(ARMS):
        out.append(_record("descriptive_curves", tol, problems=[f"descriptive_curves: arms "
                                                                f"{sorted(ec) if isinstance(ec, dict) else ec!r}"]))
    else:
        for arm in ARMS:
            y = ec[arm]
            if not isinstance(y, dict) or sorted(y) != ["control", "excess", "own"]:
                out.append(_record(f"descriptive_curves.{arm}", tol, problems=[f"{arm}: keys"]))
                continue
            diffs, probs = [], []
            for key in ("own", "control", "excess"):
                _walk_numeric(rebuilt["descriptive_curves"][arm][key], y[key], f"descriptive_curves.{arm}.{key}",
                              diffs, probs)
            out.append(_record(f"descriptive_curves.{arm}", tol, diffs, probs))
    return out


# ----------------------------------------------------------------------------- pipeline

def run_pipeline(pop, arrays, expected, tol, band=BAND, seeds=None, load_failures=()):
    pop_failures, notes, items = validate_population(pop)
    failures = list(load_failures) + pop_failures + validate_readouts(arrays)
    res = {"input_checks": {"passed": not failures, "failures": failures, "notes": notes},
           "rebuilt": None, "comparisons": None, "internal": None}
    if failures:
        res["status"] = "input_rejected"
        return res
    res["rebuilt"], res["internal"] = rebuild(items, arrays, band, seeds)
    if expected is not None:
        res["comparisons"] = compare(res["rebuilt"], expected, tol)
        res["status"] = "pass" if all(c["passed"] for c in res["comparisons"]) else "fail"
    else:
        res["status"] = "rebuilt_no_comparison"
    return res


def load_readouts(readout_dir):
    arrays, failures, hashes = {}, [], {}
    for arm in ARMS:
        path = Path(readout_dir) / f"{arm}.npz"
        if not path.is_file():
            failures.append(f"{arm}: missing file {path}")
            continue
        hashes[arm] = {"path": str(path), "sha256": sha256_file(path)}
        with np.load(path, allow_pickle=False) as z:
            if "allrank" not in z.files:
                failures.append(f"{arm}: no 'allrank' key in {path}")
                continue
            arrays[arm] = z["allrank"]
    return arrays, failures, hashes


def check_files(population, readouts, expected_path, tol, band=BAND, seeds=None):
    pop = json.loads(Path(population).read_text())
    arrays, load_failures, rd_hashes = load_readouts(readouts)
    expected = json.loads(Path(expected_path).read_text()) if expected_path else None
    res = run_pipeline(pop, arrays, expected, tol, band, seeds, load_failures)
    inputs = {"population": {"path": str(Path(population).resolve()), "sha256": sha256_file(population)},
              "readouts": rd_hashes}
    if expected_path:
        inputs["expected_analysis"] = {"path": str(Path(expected_path).resolve()), "sha256": sha256_file(expected_path)}
    return res, inputs


def header(mode, tol):
    return {"schema": "independent_endpoint_check.v1", "mode": mode,
            "checker": {"path": str(CHECKER_PATH), "sha256": sha256_file(CHECKER_PATH)},
            "python": platform.python_version(), "numpy": np.__version__,
            "parameters": {"band": list(BAND), "top_k": TOPK, "replicates": N_REPLICATES, "chunk": CHUNK,
                           "seeds": SEEDS, "tolerance": tol, "arms_columns": N_COLUMNS}}


# ----------------------------------------------------------------------------- self-test

def ensure_base_fixtures(base):
    manifest_path = base / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        bad = [rel for rel, rec in manifest["files"].items() if sha256_file(base / rel) != rec["sha256"]]
        return manifest, bad, "verified_existing_fixtures"
    if base.exists() and any(base.rglob("*")):
        raise RuntimeError(f"{base} exists without MANIFEST.json; refusing to modify it")
    (base / "readouts").mkdir(parents=True, exist_ok=True)
    files = {}
    for rel, src in (("population.json", SOURCE_RESULTS / "population.json"),
                     ("expected_analysis.json", SOURCE_ANALYSIS)):
        data = src.read_bytes()
        with open(base / rel, "xb") as fh:
            fh.write(data)
        files[rel] = {"source": str(src), "source_sha256": hashlib.sha256(data).hexdigest(),
                      "sha256": sha256_file(base / rel), "note": "byte copy"}
    for arm in ARMS:
        src = SOURCE_RESULTS / "readouts" / f"{arm}.npz"
        with np.load(src, allow_pickle=False) as z:
            a = z["allrank"]
        rel = f"readouts/{arm}.npz"
        with open(base / rel, "xb") as fh:
            np.savez_compressed(fh, allrank=a)
        with np.load(base / rel, allow_pickle=False) as z:
            if array_sha256(z["allrank"]) != array_sha256(a):
                raise RuntimeError(f"fixture copy of {arm} differs from source")
        files[rel] = {"source": str(src), "source_sha256": sha256_file(src), "sha256": sha256_file(base / rel),
                      "allrank_sha256": array_sha256(a), "note": "allrank key only"}
    manifest = {"created_by": str(CHECKER_PATH), "files": files}
    write_json_exclusive(manifest_path, manifest)
    return manifest, [], "created_fixtures"


def summarize(res):
    s = {"status": res["status"], "input_failures": res["input_checks"]["failures"][:10]}
    if res["comparisons"] is not None:
        s["failed_comparisons"] = [c["quantity"] for c in res["comparisons"] if not c["passed"]]
        s["comparisons"] = {c["quantity"]: {"passed": c["passed"], "max_abs_diff": c["max_abs_diff"],
                                            "n_problems": c["n_problems"],
                                            "first_problem": c["problems"][0] if c["problems"] else None}
                            for c in res["comparisons"]}
    return s


def vectors(rebuilt):
    return {"estimates": np.array(rebuilt["estimates"]),
            "secondary": np.array([s["estimate"] for s in rebuilt["secondary_family"]]),
            "curves": {a: {k: np.array(rebuilt["descriptive_curves"][a][k]) for k in ("own", "control", "excess")}
                       for a in ARMS}}


def predicted_zero():
    return {"estimates": np.zeros(7), "secondary": np.zeros(20),
            "curves": {a: {k: np.zeros(N_COLUMNS[a]) for k in ("own", "control", "excess")} for a in ARMS}}


def prediction_error(obs, base, pred):
    errs = [np.max(np.abs((obs["estimates"] - base["estimates"]) - pred["estimates"])),
            np.max(np.abs((obs["secondary"] - base["secondary"]) - pred["secondary"]))]
    for a in ARMS:
        for k in ("own", "control", "excess"):
            errs.append(np.max(np.abs((obs["curves"][a][k] - base["curves"][a][k]) - pred["curves"][a][k])))
    return float(max(errs))


def predict_hit_change(items, arrays, arm, pos, name, col, new_rank, band=BAND):
    """Analytic change of estimates, secondary estimates and curves for one changed rank."""
    n, band = len(items), list(band)
    pred = predicted_zero()
    old_rank = int(arrays[arm][pos, name, col])
    dhit = float(0 <= new_rank < TOPK) - float(0 <= old_rank < TOPK)
    item = next((it for it in items if it.position == pos), None)
    if item is None or dhit == 0:
        return pred, {"dhit": dhit}
    S = item.slots
    a_own = sum(1.0 for o, _ in S if o == name) / len(S)
    a_ctl = sum((1.0 / len(c)) if name in c else 0.0 for _, c in S) / len(S)
    dI, dC = dhit * a_own, dhit * a_ctl
    dE = dI - dC
    pred["curves"][arm]["own"][col] = dI / n
    pred["curves"][arm]["control"][col] = dC / n
    pred["curves"][arm]["excess"][col] = dE / n
    inb = col in band
    dIB, dCB, dEB = (dI / len(band), dC / len(band), dE / len(band)) if inb else (0.0, 0.0, 0.0)
    if arm == "fit01":
        pred["estimates"] += np.array([dEB, dIB, dCB, 0, 0, dIB, dCB]) / n
    if arm == "raw":
        pred["estimates"] += np.array([-dEB, 0, 0, dIB, dCB, -dIB, -dCB]) / n
    for k, sp in enumerate(secondary_specs(band)):
        sign = (arm == sp["left"]) - (arm == sp["right"])
        R = sp["region"]
        if sign == 0 or col not in R:
            continue
        if sp["metric"] == "fixed_mean":
            pred["secondary"][k] = sign * dE / len(R) / n
        else:
            h = (arrays[arm][pos, name, R] >= 0) & (arrays[arm][pos, name, R] < TOPK)
            h2 = h.copy()
            h2[R.index(col)] = 0 <= new_rank < TOPK
            pred["secondary"][k] = sign * (float(h2.any()) - float(h.any())) * (a_own - a_ctl) / n
    return pred, {"dhit": dhit, "slot_share_as_intended": a_own, "slot_share_as_control": a_ctl}


def self_test(out_path, tol):
    base_dir = FIXTURE_DIR / "base"
    manifest, bad_fixtures, how = ensure_base_fixtures(base_dir)
    doc = header("self_test", tol)
    doc["fixtures"] = {"base_dir": str(base_dir), "how": how, "hash_mismatches": bad_fixtures, "manifest": manifest}
    cases = []

    def case(cid, description, res, assertions, perturbation=None, notes=None):
        passed = all(a["passed"] for a in assertions)
        cases.append({"id": cid, "description": description, "perturbation": perturbation, "notes": notes or [],
                      "observed": summarize(res), "assertions": assertions, "passed": passed})
        print(f"{cid}: {'PASS' if passed else 'FAIL'} ({res['status']})", file=sys.stderr)

    def A(name, ok, **detail):
        return {"name": name, "passed": bool(ok), **detail}

    def failed(res):
        return res["status"] == "fail"

    def rejected(res, needle):
        return res["status"] == "input_rejected" and any(needle in f for f in res["input_checks"]["failures"])

    def cmp_max(res, q):
        return next(c["max_abs_diff"] for c in res["comparisons"] if c["quantity"] == q)

    res0, inputs0 = check_files(base_dir / "population.json", base_dir / "readouts",
                                base_dir / "expected_analysis.json", tol)
    case("U0", "unperturbed fixture copy vs saved analysis (file-loading path)", res0,
         [A("fixture hashes match manifest", not bad_fixtures), A("status pass", res0["status"] == "pass")],
         perturbation={"inputs": inputs0})
    if res0["status"] != "pass":
        doc["cases"], doc["passed"] = cases, False
        write_json_exclusive(out_path, doc)
        return 1

    expected = json.loads((base_dir / "expected_analysis.json").read_text())
    pop0 = json.loads((base_dir / "population.json").read_text())
    arrays0, _, _ = load_readouts(base_dir / "readouts")
    items0 = res0["internal"]["items"]
    X0, Xs0 = res0["internal"]["X"], res0["internal"]["Xs"]
    v0 = vectors(res0["rebuilt"])
    sec_ids = [s["id"] for s in res0["rebuilt"]["secondary_family"]]
    names = pop0["names"]["multihop"]
    n = len(items0)

    def run(pop=pop0, arrays=arrays0, band=BAND, seeds=None):
        return run_pipeline(pop, arrays, expected, tol, band, seeds)

    def with_array(arm, a):
        d = dict(arrays0)
        d[arm] = a
        return d

    def flip_case(cid, description, arm, cand, new_rank, extra_asserts, notes=None):
        pos, slot, name, col = cand
        a = arrays0[arm].copy()
        old = int(a[pos, name, col])
        a[pos, name, col] = new_rank
        res = run(arrays=with_array(arm, a))
        pred, pinfo = predict_hit_change(items0, arrays0, arm, pos, name, col, new_rank)
        v = vectors(res["rebuilt"]) if res["rebuilt"] else None
        err = prediction_error(v, v0, pred) if v else None
        obs_est = (v["estimates"] - v0["estimates"]).tolist() if v else None
        obs_sec = dict(zip(sec_ids, (v["secondary"] - v0["secondary"]).tolist())) if v else None
        asserts = [A("input accepted", res["input_checks"]["passed"]),
                   A("comparison against unperturbed expectation fails", failed(res)),
                   A("all observed deltas (7 estimates, 20 secondary estimates, all curves) equal analytic prediction",
                     err is not None and err <= tol, max_abs_prediction_error=err)]
        asserts += extra_asserts(v, pred)
        pert = {"arm": arm, "item_position": pos, "item_name": pop0["rows"][pos].get("name"), "slot": slot,
                "name_index": name, "name": names[name], "virtual_column": col, "old_rank": old,
                "new_rank": new_rank, **pinfo,
                "predicted_estimate_deltas": dict(zip(VALUE_ORDER, pred["estimates"].tolist())),
                "observed_estimate_deltas": dict(zip(VALUE_ORDER, obs_est)) if obs_est else None,
                "predicted_secondary_deltas_nonzero": {i: d for i, d in zip(sec_ids, pred["secondary"].tolist()) if d},
                "observed_secondary_deltas_nonzero": {i: d for i, d in obs_sec.items() if d} if obs_sec else None}
        case(cid, description, res, asserts, pert, notes)

    # F1: fit01 intended rank 9 -> 10 inside B
    c1 = [(it.position, s, o, c) for it in items0 for s, (o, _) in enumerate(it.slots) for c in BAND
          if arrays0["fit01"][it.position, o, c] == 9]
    flip_case("F1", "fit01 intended rank 9 -> 10 at one column inside B for one item", "fit01", c1[0], 10,
              lambda v, p: [A("primary delta == -1/(|S| * 12 * N)", abs((v["estimates"][0] - v0["estimates"][0])
                                                                        - (-1.0 / (len(items0[[it.position for it in items0].index(c1[0][0])].slots) * 12 * n))) <= tol,
                              predicted=p["estimates"][0], observed=float(v["estimates"][0] - v0["estimates"][0]))],
              notes=[f"{len(c1)} candidate (item, column) pairs in B; first in item/slot/column order used"])

    # F2: the same flip at virtual 100
    c2 = [(it.position, s, o, 100) for it in items0 for s, (o, _) in enumerate(it.slots)
          if arrays0["fit01"][it.position, o, 100] == 9]
    f2_notes = []
    if not c2:
        hits100 = [(int(arrays0["fit01"][it.position, o, 100]), it.position, s, o) for it in items0
                   for s, (o, _) in enumerate(it.slots) if 0 <= arrays0["fit01"][it.position, o, 100] < TOPK]
        best = max(r for r, *_ in hits100)
        r, pos, s, o = next(h for h in hits100 if h[0] == best)
        c2 = [(pos, s, o, 100)]
        f2_notes.append("SPEC/DATA DISAGREEMENT: no eligible item has fit01 intended rank exactly 9 at virtual 100. "
                        f"Used the intended hit with the largest rank there (rank {r}) set to 10; the hit arithmetic "
                        f"is identical. Intended hits at virtual 100 (rank, item, slot, name): {hits100}")
    loop3 = [sec_ids.index("early_loop3_fixed_mean"), sec_ids.index("early_loop3_any_layer")]
    pos2, _, o2, _ = c2[0]
    n_hits_loop3 = int(((arrays0["fit01"][pos2, o2, 96:144] >= 0) & (arrays0["fit01"][pos2, o2, 96:144] < TOPK)).sum())
    f2_notes.append(f"item {pos2} has {n_hits_loop3} fit01 intended hits in virtual 96..143; any_layer changes only if 1")
    flip_case("F2", "fit01 intended hit flipped to non-hit at virtual 100", "fit01", c2[0], 10,
              lambda v, p: [A("all 7 estimates bitwise unchanged", np.array_equal(v["estimates"], v0["estimates"])),
                            A("early_loop3_fixed_mean changes by predicted nonzero amount",
                              p["secondary"][loop3[0]] != 0 and abs((v["secondary"][loop3[0]] - v0["secondary"][loop3[0]])
                                                                    - p["secondary"][loop3[0]]) <= tol,
                              predicted=p["secondary"][loop3[0]],
                              observed=float(v["secondary"][loop3[0]] - v0["secondary"][loop3[0]])),
                            A("early_loop3_any_layer delta equals prediction", abs((v["secondary"][loop3[1]] - v0["secondary"][loop3[1]])
                                                                                  - p["secondary"][loop3[1]]) <= tol,
                              predicted=p["secondary"][loop3[1]],
                              observed=float(v["secondary"][loop3[1]] - v0["secondary"][loop3[1]]))],
              notes=f2_notes)

    # F3: band 170..181
    res = run(band=range(170, 182))
    case("F3", "band evaluated as virtual 170..181", res,
         [A("comparison fails", failed(res)), A("estimates comparison fails", cmp_max(res, "estimates") > tol),
          A("secondary identity (local regions) mismatches",
            not next(c for c in res["comparisons"] if c["quantity"] == "secondary_family.identity")["passed"])],
         perturbation={"band": list(range(170, 182))})

    # F4: one control removed from one slot
    pop = copy.deepcopy(pop0)
    removed = pop["rows"][0]["control_indices"][0].pop()
    res = run(pop=pop)
    case("F4", "one control removed from row 0 slot 0", res, [A("input rejected", rejected(res, "row 0 slot 0"))],
         perturbation={"row": 0, "slot": 0, "removed_control_index": removed})

    # F5a: row appended as duplicate (shape change)
    pop = copy.deepcopy(pop0)
    pop["rows"].append(copy.deepcopy(pop0["rows"][0]))
    arrays = {arm: np.concatenate([arrays0[arm], arrays0[arm][:1]], axis=0) for arm in ARMS}
    res = run(pop=pop, arrays=arrays)
    case("F5a", "row 0 appended as a 161st item (population and all arms)", res,
         [A("input rejected", rejected(res, "rows must be a list of 160"))], perturbation={"duplicated_row": 0})

    # F5b: duplicate one item over another (160 rows, weighting change)
    pos_of = [it.position for it in items0]
    dst = pos_of[-1]
    src = next(p for p in pos_of if X0[pos_of.index(p), 0] != X0[-1, 0])
    pop = copy.deepcopy(pop0)
    pop["rows"][dst] = copy.deepcopy(pop0["rows"][src])
    arrays = {}
    for arm in ARMS:
        a = arrays0[arm].copy()
        a[dst] = a[src]
        arrays[arm] = a
    res = run(pop=pop, arrays=arrays)
    ri, rd = pos_of.index(src), pos_of.index(dst)
    pred = predicted_zero()
    pred["estimates"] = (X0[ri] - X0[rd]) / n
    pred["secondary"] = (Xs0[ri] - Xs0[rd]) / n
    for arm in ARMS:
        for k in ("own", "control", "excess"):
            cv = res0["internal"]["curves"][arm][k]
            pred["curves"][arm][k] = (cv[ri] - cv[rd]) / n
    err = prediction_error(vectors(res["rebuilt"]), v0, pred)
    case("F5b", "item row duplicated over the last item (population row and all arms): item weight doubled", res,
         [A("input accepted", res["input_checks"]["passed"]), A("comparison fails", failed(res)),
          A("estimate/secondary/curve deltas equal (x_src - x_dst)/N", err <= tol, max_abs_prediction_error=err)],
         perturbation={"source_position": src, "overwritten_position": dst,
                       "predicted_primary_delta": float(pred["estimates"][0])})

    # F6: one item's group label changed to another existing group
    g0 = items0[0].group
    sizes = {g: sum(it.group == g for it in items0) for g in dict.fromkeys(it.group for it in items0)}
    mover = next(it for it in items0 if it.group != g0 and sizes[it.group] >= 2)
    pop = copy.deepcopy(pop0)
    pop["rows"][mover.position]["dependency_group_id"] = g0
    res = run(pop=pop)
    v = vectors(res["rebuilt"])
    case("F6", "one item's dependency_group_id changed to another existing group", res,
         [A("comparison fails", failed(res)),
          A("7 estimates bitwise unchanged", np.array_equal(v["estimates"], v0["estimates"])),
          A("secondary estimates bitwise unchanged", np.array_equal(v["secondary"], v0["secondary"])),
          A("group count unchanged", res["rebuilt"]["groups"] == res0["rebuilt"]["groups"]),
          A("family intervals changed", cmp_max(res, "family_percentile_95_intervals") > tol),
          A("secondary intervals changed", cmp_max(res, "secondary_family.simultaneous_95_interval") > tol),
          A("item-resampling interval bitwise unchanged", res["rebuilt"]["primary_item_resampling_95_interval"]
            == res0["rebuilt"]["primary_item_resampling_95_interval"])],
         perturbation={"item_position": mover.position, "old_group": mover.group, "new_group": g0})

    # F7: penultimate columns shifted by one
    a = arrays0["penultimate"].copy()
    a[:, :, 1:] = arrays0["penultimate"][:, :, :-1]
    res = run(arrays=with_array("penultimate", a))
    v = vectors(res["rebuilt"])
    pen = [k for k, i in enumerate(sec_ids) if "penultimate" in i]
    other = [k for k in range(20) if k not in pen]
    case("F7", "penultimate allrank shifted by one column (new[c] = old[c-1], column 0 kept)", res,
         [A("comparison fails", failed(res)),
          A("every penultimate contrast estimate changed", bool(np.all(np.abs(v["secondary"][pen] - v0["secondary"][pen]) > tol)),
            deltas={sec_ids[k]: float(v["secondary"][k] - v0["secondary"][k]) for k in pen}),
          A("non-penultimate contrast estimates bitwise unchanged", np.array_equal(v["secondary"][other], v0["secondary"][other])),
          A("7 estimates bitwise unchanged", np.array_equal(v["estimates"], v0["estimates"])),
          A("penultimate curves comparison fails", cmp_max(res, "descriptive_curves.penultimate") > tol)],
         perturbation={"arm": "penultimate", "shift": "+1 column"},
         notes=["the shared max-t quantile also moves, so other contrasts' simultaneous intervals change too"])

    # F8: one padding entry set to 3
    a = arrays0["raw"].copy()
    a[0, N_NAMES, 0] = 3
    res = run(arrays=with_array("raw", a))
    case("F8", "raw allrank[0, 80, 0] (padding) set to 3", res, [A("input rejected", rejected(res, "padding"))],
         perturbation={"arm": "raw", "index": [0, N_NAMES, 0], "new_value": 3})

    # F9: bootstrap seeds changed
    for sub, key, quantity in (("F9a", "family", "family_percentile_95_intervals"),
                               ("F9b", "item", "primary_item_resampling_95_interval"),
                               ("F9c", "secondary", "secondary_family.simultaneous_95_interval")):
        seeds = dict(SEEDS)
        seeds[key] = 2026090999
        res = run(seeds=seeds)
        untouched = [q for k2, q in (("family", "family_percentile_95_intervals"),
                                     ("item", "primary_item_resampling_95_interval"),
                                     ("secondary", "secondary_family.simultaneous_95_interval")) if k2 != key]
        case(sub, f"{key} bootstrap seed changed to 2026090999", res,
             [A("comparison fails", failed(res)), A(f"{quantity} differs", cmp_max(res, quantity) > tol),
              A("estimates identical", cmp_max(res, "estimates") <= tol),
              A("other interval families still pass", all(cmp_max(res, q) <= tol for q in untouched))],
             perturbation={"seeds": seeds})

    # F10: one raw control rank 10 -> 9 inside B
    c10 = [(it.position, s, int(k), c) for it in items0 for s, (_, ctrl) in enumerate(it.slots) for c in BAND
           for k in ctrl if arrays0["raw"][it.position, k, c] == 10]
    it10 = items0[pos_of.index(c10[0][0])]
    d_ctl = 1.0 / (len(it10.slots) * len(it10.slots[c10[0][1]][1]) * len(BAND) * n)
    flip_case("F10", "raw control rank 10 -> 9 at one column inside B for one item", "raw", c10[0], 9,
              lambda v, p: [A("raw_control delta == +1/(|S| * |controls| * 12 * N)",
                              abs((v["estimates"][4] - v0["estimates"][4]) - d_ctl) <= tol,
                              predicted=d_ctl, observed=float(v["estimates"][4] - v0["estimates"][4])),
                            A("control_difference delta == -that amount",
                              abs((v["estimates"][6] - v0["estimates"][6]) + d_ctl) <= tol,
                              observed=float(v["estimates"][6] - v0["estimates"][6]))],
              notes=[f"{len(c10)} candidate (item, slot, control, column) tuples; first in item/slot/column/name order used"])

    doc["cases"] = cases
    doc["passed"] = all(c["passed"] for c in cases)
    write_json_exclusive(out_path, doc)
    return 0 if doc["passed"] else 1


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population")
    ap.add_argument("--readouts")
    ap.add_argument("--expected-analysis")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-12)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        print(f"refusing to overwrite {args.out}", file=sys.stderr)
        return 2
    if args.self_test:
        if args.population or args.readouts or args.expected_analysis:
            ap.error("--self-test takes only --out and --tolerance")
        return self_test(Path(args.out), args.tolerance)
    if not (args.population and args.readouts):
        ap.error("--population and --readouts are required unless --self-test")
    res, inputs = check_files(args.population, args.readouts, args.expected_analysis, args.tolerance)
    doc = header("check", args.tolerance)
    doc.update({"inputs": inputs, "status": res["status"], "input_checks": res["input_checks"],
                "comparisons": res["comparisons"], "rebuilt": res["rebuilt"]})
    write_json_exclusive(args.out, doc)
    print(f"status: {res['status']}", file=sys.stderr)
    for c in res["comparisons"] or []:
        print(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['quantity']}: max_abs_diff={c['max_abs_diff']} "
              f"problems={c['n_problems']}", file=sys.stderr)
    for f in res["input_checks"]["failures"]:
        print(f"  INPUT FAIL: {f}", file=sys.stderr)
    return 0 if res["status"] in ("pass", "rebuilt_no_comparison") else 1


if __name__ == "__main__":
    sys.exit(main())
