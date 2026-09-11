"""Render one figure from the saved statistics of a sealed Ouro analysis.

No scoring, fitting, resampling or layer selection is performed. Example:
  python -B plot_refit_curves.py --analysis /path/to/sealed-analysis \
      --output /path/to/new-curve-figure
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_refits as refits

IO = refits.io
SCHEMA = "ouro_refit_curve_figure.v1"
STEM = "refit_layerwise_curves"
TASK_LABELS = {"multihop": "Multihop", "order-ops": "Order of operations"}
FIT_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
MEAN_COLOR = "#252525"
INTERVAL_COLOR = "#536878"
INTERVAL_MODE = "crossed_fit_item"
INTERVAL_FIELD = "pointwise95"


def read_curves(report):
    """Check plotting geometry; retain the producer's values without recomputation."""
    np = refits.numpy()
    if (report.get("schema") != refits.SCHEMA
            or not refits._same(report.get("fit_ids"), refits.FIT_IDS)
            or report.get("uncertainty", {}).get("pointwise") != "percentile95"):
        raise ValueError("report identity or pointwise uncertainty policy differs")
    tasks = report["P1"]["tasks"]
    if set(tasks) != set(refits.TASKS):
        raise ValueError("P1 must contain the two frozen tasks")
    curves = {}
    for task in refits.TASKS:
        cells = tasks[task]["cells"]
        values = {
            "per_fit": np.asarray(cells["per_fit"], dtype=np.float64),
            "mean": np.asarray(cells["mean"], dtype=np.float64),
            "interval": np.asarray(
                cells["intervals"][INTERVAL_MODE][INTERVAL_FIELD], dtype=np.float64),
        }
        for name, shape in (("per_fit", (5, 192)), ("mean", (192,)),
                            ("interval", (192, 2))):
            if values[name].shape != shape or not np.isfinite(values[name]).all():
                raise ValueError(f"{task} {name} must be finite with shape {shape}")
        if np.any(values["interval"][:, 0] > values["interval"][:, 1]):
            raise ValueError(f"{task} contains a reversed pointwise interval")
        curves[task] = values
    return curves


def make_figure(curves):
    """Plot all 192 saved cells with one y scale across both tasks and all loops."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    np = refits.numpy()
    # Extrema are used only to set display limits, never to select a panel/cell.
    lower, upper = 0.0, 0.0
    for values in curves.values():
        for array in values.values():
            lower = min(lower, float(array.min()))
            upper = max(upper, float(array.max()))
    margin = (upper - lower) * 0.06 if upper > lower else 0.05
    limits = [lower - margin, upper + margin]
    x = np.arange(1, 49)
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.765,
                        hspace=0.28, wspace=0.13)
    for row, task in enumerate(refits.TASKS):
        values = curves[task]
        for loop in range(4):
            ax = axes[row, loop]
            selected = slice(48 * loop, 48 * (loop + 1))
            interval = values["interval"][selected]
            ax.fill_between(x, interval[:, 0], interval[:, 1],
                            color=INTERVAL_COLOR, alpha=0.18, linewidth=0, zorder=1)
            ax.axhline(0, color="#777777", linewidth=0.8, zorder=2)
            for position, color in enumerate(FIT_COLORS):
                ax.plot(x, values["per_fit"][position, selected],
                        color=color, linewidth=0.95, alpha=0.72, zorder=3)
            ax.plot(x, values["mean"][selected], color=MEAN_COLOR,
                    linewidth=2.0, zorder=4)
            if loop == 3:
                ax.plot(48, values["mean"][191], marker="D", markersize=5.5,
                        markerfacecolor="white", markeredgecolor=MEAN_COLOR,
                        linestyle="none", clip_on=False, zorder=5)
            ax.set(xlim=(1, 48), ylim=limits, xticks=[1, 12, 24, 36, 48],
                   title=f"Loop {loop + 1}")
            ax.tick_params(labelsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel("Physical layer (1–48)", fontsize=9)
            if loop == 0:
                ax.set_ylabel(f"{TASK_LABELS[task]}\nΔ excess hit@10", fontsize=10)

    fig.suptitle("Ouro: J-Lens − raw logit lens across five N100 refits",
                 y=0.985, fontsize=14)
    fig.text(0.5, 0.942,
             "95% pointwise intervals from crossed resampling of fits and items "
             "(not simultaneous)", ha="center", fontsize=10)
    fit_handles = [Line2D([], [], color=color, linewidth=1.4, label=f"Fit {fit_id}")
                   for fit_id, color in zip(refits.FIT_IDS, FIT_COLORS, strict=True)]
    fig.legend(handles=fit_handles, loc="upper center", bbox_to_anchor=(0.5, 0.918),
               ncol=5, frameon=False, fontsize=9)
    fig.legend(handles=[
        Line2D([], [], color=MEAN_COLOR, linewidth=2, label="Saved fit mean"),
        Patch(facecolor=INTERVAL_COLOR, alpha=0.18, label="Pointwise 95% interval"),
        Line2D([], [], color=MEAN_COLOR, marker="D", markerfacecolor="white",
               linestyle="none", label="Identity: loop 4, layer 48 (v191)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.871), ncol=3,
        frameon=False, fontsize=9)
    fig.text(0.5, 0.035,
             "All 192 virtual cells are displayed. The same y scale is used in all panels.",
             ha="center", fontsize=9, color="#555555")
    return fig, limits


def render(analysis_root, output):
    root, out = IO._no_links(analysis_root), IO._no_links(output)
    if out == root or root in out.parents:
        raise ValueError("figure output must be outside the sealed analysis directory")
    if out.exists():
        raise FileExistsError(f"figure output must be a new directory: {out}")

    inputs = {name: IO._record(root / name)
              for name in ("report.json", "OWNER.json", "COMPLETE.json")}
    sources = {str(path): IO._record(path) for path in
               (Path(__file__).resolve(), Path(refits.__file__).resolve(),
                Path(IO.__file__).resolve())}
    checked_before = refits.validate_output(root)
    for name, record in inputs.items():
        IO._verify_record(root / name, record)
    report = IO._json(root / "report.json")
    curves = read_curves(report)
    IO._verify_record(root / "report.json", inputs["report.json"])

    # Keep matplotlib's font/config cache out of the sealed input and figure output.
    old_config = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="ouro-refit-plot-mpl-") as config:
        os.environ["MPLCONFIGDIR"] = config
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                                        "path.simplify": False, "pdf.fonttype": 42}):
                fig, limits = make_figure(curves)
                try:
                    rendered = {}
                    for kind in ("png", "pdf"):
                        buffer = BytesIO()
                        fig.savefig(buffer, format=kind, dpi=180)
                        rendered[f"{STEM}.{kind}"] = buffer.getvalue()
                finally:
                    plt.close(fig)
            matplotlib_version = matplotlib.__version__
        finally:
            if old_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_config

    checked_after = refits.validate_output(root)
    for name, record in inputs.items():
        IO._verify_record(root / name, record)
    for path, record in sources.items():
        IO._verify_record(Path(path), record)
    out.mkdir(parents=True, exist_ok=False)
    IO._no_links(out)
    for name, data in rendered.items():
        with IO._no_links(out / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    # Publish provenance last; partial writes never claim a completed figure.
    for name, record in inputs.items():
        IO._verify_record(root / name, record)
    provenance = {
        "schema": SCHEMA,
        "created_utc": IO._utc_now().isoformat(),
        "purpose": "presentation only; no new scores, fitting, resampling or selection",
        "analysis_root": str(root), "input_files": inputs,
        "implementation_sources": sources,
        "validation": {"before": checked_before, "after": checked_after,
                       "report_owner_and_seal_unchanged": True},
        "figure": {
            "tasks": list(refits.TASKS), "fit_ids": refits.FIT_IDS,
            "source": "report.json:P1.tasks[task].cells",
            "curves": ["per_fit", "mean"],
            "interval_source": f"intervals.{INTERVAL_MODE}.{INTERVAL_FIELD}",
            "interval_interpretation": "pointwise crossed-fit/item 95%; not simultaneous",
            "layout": "two task rows by four loop columns",
            "physical_x_limits": [1, 48], "virtual_indices": [0, 191],
            "y_scale": "shared across both tasks and all cells, curves and interval bounds",
            "y_limits": limits, "identity_virtual_index": 191,
            "historical_region_shading": False, "smoothing": False,
        },
        "runtime": {"python": sys.version, "numpy": refits.numpy().__version__,
                    "matplotlib": matplotlib_version},
        "outputs": {name: IO._record(out / name) for name in rendered},
    }
    IO._new_json(out / "provenance.json", provenance)
    return {"status": "passed", "output": str(out),
            "files": sorted([*rendered, "provenance.json"]),
            "input_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="complete sealed output of analyze_refits.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory outside the sealed analysis")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.analysis, args.output), indent=2))


if __name__ == "__main__":
    main()
