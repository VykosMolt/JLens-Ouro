"""Retain a uniform random sample, chosen without inspecting readout outcomes."""
from pathlib import Path

import numpy as np

from analyze_followup import MAIN, METHODS, TASKS, eligible_slots, load_run, save_json, score_items


def main():
    items, names, arrays = load_run(MAIN)
    scores = {task: score_items(items, names, arrays, task) for task in TASKS}
    eligible = [i for i, item in enumerate(items) if eligible_slots(item)]
    selected = np.random.default_rng(20260907).choice(eligible, 10, replace=False)
    rows = []
    for i in selected:
        item = items[i]
        row = {k: item[k] for k in ("name", "task", "prompt", "intermediates", "target", "continuation", "correct", "exit_top1")}
        row["index"] = int(i)
        row["eligible_slots"] = eligible_slots(item)
        row["methods"] = {}
        for method in METHODS:
            j = np.flatnonzero(scores[item["task"]]["indices"] == i).item()
            s = scores[item["task"]][method]
            row["methods"][method] = {k: s[k][j] for k in ("any_hit", "any_control", "any_excess")}
            row["methods"][method]["ordinary_best_ranks_by_slot_and_loop"] = [
                arrays[method+"_rank"][i, slot].reshape(4, 48).min(axis=1) + 1
                for slot in row["eligible_slots"]
            ]
        rows.append(row)
    out = Path(__file__).resolve().parent / "results"
    save_json(out / "random_examples.json", {"seed": 20260907, "population": len(eligible), "sampling": "10 items uniformly without replacement; no outcome filtering", "items": rows})
    text = ["# Random examples from the main N=100 evaluation", "", "Ten of 141 eligible items, drawn uniformly without replacement with NumPy seed 20260907. Nothing is filtered for model or lens success. Ranks below are ordinary one-based ranks, minimized across each loop's 48 layers; they are descriptive oracle summaries. Trailing prompt whitespace is trimmed only in this display. Exact prompts, exits, scores and ranks are in `random_examples.json`.", ""]
    for row in rows:
        text += ["## " + row["name"], "", "```text", row["prompt"].rstrip(), "```", "", f"Task labels: {row['intermediates']}; target: {row['target']!r}; continuation: {row['continuation']!r}; historical pass: {row['correct']}.", ""]
        for method, values in row["methods"].items():
            ranks = [x.tolist() for x in values["ordinary_best_ranks_by_slot_and_loop"]]
            excess = [round(float(x), 3) for x in values["any_excess"]]
            text.append(f"- {method}: best ranks by eligible slot, loops 1–4: {ranks}; excess hit@10: {excess}.")
        text.append("")
    (out / "random_examples.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
