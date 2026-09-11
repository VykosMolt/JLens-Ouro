"""Known-intermediate evaluation stimuli (Anthropic's lens-eval sets) for Ouro.

Each intermediate is expanded to surface forms (README convention: numbers ->
digit and word forms, operations -> symbol and word forms), tokenized with and
without a leading space, and kept only if it is a single Ouro token. Readout
position is the last prompt token (the token immediately preceding `target`).
An intermediate is *leaked* if any of its single-token forms occurs in the
prompt's token ids.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

def _jlens_data() -> Path:
    """Anthropic's eval stimuli live in the repo, not the package, so locate them from the
    editable install rather than assuming $HOME (which differs on a rented pod)."""
    import jlens

    for c in (Path(jlens.__file__).resolve().parents[1], Path.home() / "jacobian-lens"):
        if (c / "data" / "evaluations").is_dir():
            return c / "data" / "evaluations"
    raise FileNotFoundError("jacobian-lens data/evaluations not found; clone the repo and pip install -e it")


JLENS_DATA = _jlens_data()

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 24: "twenty-four", 26: "twenty-six", 50: "fifty",
    52: "fifty-two",
}
WORD_NUMBERS = {v: k for k, v in NUMBER_WORDS.items()}
ORDINALS = {"third": 3}
OPERATIONS = {
    "addition": ["+", "plus", "add", "added", "sum"],
    "subtraction": ["-", "minus", "subtract", "subtracted", "difference"],
    "multiplication": ["*", "×", "times", "multiply", "multiplied", "product"],
    "division": ["/", "//", "÷", "divided", "divide", "quotient"],
    "mod": ["%", "modulo", "remainder"],
    "squared": ["^", "**", "square", "squared"],
}


def surface_forms(intermediate: str) -> list[str]:
    forms = [intermediate]
    if intermediate.isdigit():
        n = int(intermediate)
        if n in NUMBER_WORDS:
            forms.append(NUMBER_WORDS[n])
    elif intermediate in WORD_NUMBERS:
        forms.append(str(WORD_NUMBERS[intermediate]))
    elif intermediate in ORDINALS:
        forms.append(str(ORDINALS[intermediate]))
    elif intermediate in OPERATIONS:
        forms.extend(OPERATIONS[intermediate])
    out = []
    for f in forms:
        for v in (f, f.capitalize(), f.lower()):
            if v not in out:
                out.append(v)
    return out


@dataclass
class Item:
    name: str
    task: str
    prompt: str
    target: str
    intermediates: list[str]
    token_ids: list[int]
    intermediate_tokens: dict[str, list[int]]  # intermediate -> single-token ids
    leaked: dict[str, bool] = field(default_factory=dict)

    @property
    def scorable(self) -> list[str]:
        return [k for k, v in self.intermediate_tokens.items() if v]


def single_token_ids(tokenizer, forms: list[str]) -> list[int]:
    ids: list[int] = []
    for f in forms:
        for variant in (f, " " + f):
            enc = tokenizer(variant, add_special_tokens=False).input_ids
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
    return ids


def readout_context(encode, prompt: str, target: str) -> tuple[list[int], int]:
    """Token ids the model sees at readout: the common prefix of encode(prompt) and
    encode(prompt + target). The readout position is its last token, the one
    immediately preceding the target's first token (a trailing prompt space merges
    into a word target like " Atlantic" but stays a lone token before digits)."""
    a, b = encode(prompt), encode(prompt + target)
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return b[:n], len(a) - n


def load_items(tokenizer, tasks=("multihop", "order-ops"), encode=None) -> list[Item]:
    tasks = tuple(tasks)
    if (not tasks or any(not isinstance(task, str)
                         or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task)
                         for task in tasks)
            or len(tasks) != len(set(tasks))):
        raise ValueError("evaluation task names must be nonempty, unique, and path-safe")
    encode = encode or (lambda s: tokenizer(s).input_ids)
    items = []
    for task in tasks:
        document = json.loads(
            (JLENS_DATA / f"lens-eval-{task}.json").read_text(encoding="utf-8")
        )
        raw = document.get("items") if isinstance(document, dict) else None
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"evaluation task {task!r} has no item list")
        for r in raw:
            if (not isinstance(r, dict)
                    or not all(isinstance(r.get(field), str) and r[field]
                               for field in ("name", "prompt", "target"))
                    or not isinstance(r.get("intermediates"), list)
                    or not 1 <= len(r["intermediates"]) <= 3
                    or not all(isinstance(value, str) and value
                               for value in r["intermediates"])
                    or len(r["intermediates"]) != len(set(r["intermediates"]))):
                raise ValueError(f"evaluation task {task!r} contains a malformed item")
            ids, dropped = readout_context(encode, r["prompt"], r["target"])
            if (not isinstance(ids, list) or not ids
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)):
                raise ValueError(f"evaluation item {r['name']} has no valid readout context")
            if dropped > 1:
                raise ValueError(
                    f"readout boundary dropped {dropped} prompt tokens for {r['name']}; "
                    "evaluation alignment is no longer established"
                )
            inter_tokens = {}
            for inter in r["intermediates"]:
                inter_tokens[inter] = single_token_ids(tokenizer, surface_forms(inter))
            item = Item(r["name"], task, r["prompt"], r["target"], r["intermediates"], ids, inter_tokens)
            item.leaked = {k: any(t in ids for t in v) for k, v in inter_tokens.items()}
            items.append(item)
    return items


def summarize(items: list[Item], tokenizer) -> None:
    for task in sorted({i.task for i in items}):
        sub = [i for i in items if i.task == task]
        n_inter = sum(len(i.intermediates) for i in sub)
        n_scorable = sum(len(i.scorable) for i in sub)
        n_leaked = sum(i.leaked[k] for i in sub for k in i.scorable)
        print(f"{task}: {len(sub)} items, {n_inter} intermediates, {n_scorable} single-token scorable, {n_leaked} leaked")
        for i in sub:
            print(f"  readout token {tokenizer.decode([i.token_ids[-1]])!r:8s} {i.name}")
            for k in i.intermediates:
                toks = [tokenizer.decode([t]) for t in i.intermediate_tokens[k]]
                flag = "LEAK" if i.leaked.get(k) else ("    " if toks else "UNSCORABLE")
                print(f"  {flag} {i.name:32s} {k!r:18s} -> {toks}")


if __name__ == "__main__":
    import sys

    import transformers

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ouro_jlens.recurrent import OURO_SNAPSHOT

    tok = transformers.AutoTokenizer.from_pretrained(str(OURO_SNAPSHOT))
    summarize(load_items(tok), tok)
