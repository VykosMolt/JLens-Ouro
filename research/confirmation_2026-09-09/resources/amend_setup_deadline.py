"""Apply the user-authorized, one-field setup deadline amendment for attempt 05."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CONTROLLER = HERE.parent / "controller"
sys.path.insert(0, str(CONTROLLER))

import lease  # noqa: E402
import run_refits as io  # noqa: E402


EXPECTED = {
    "name": "jlens-confirm-e5b5cdde88044a96ab1410aa165a76e5",
    "pod_id": "i7zq14rj0f0ief",
    "run_id": "ouro_confirmation_20260910_fixed160_precision1",
    "status": "running",
    "setup_deadline_utc": "2026-09-10T18:20:01Z",
    "work_deadline_utc": "2026-09-10T19:20:01Z",
    "watch_deadline_utc": "2026-09-10T19:50:01Z",
    "provider_deadline_utc": "2026-09-10T20:00:01Z",
    "job_incremental_cap_usd": 3.77,
    "combined_cap_usd": 25.0,
}
NEW_SETUP_DEADLINE = "2026-09-10T18:35:01Z"
AUTHORIZATION = {
    "quote": "Extend the deadline",
    "history_window_id": "01a088be-2169-7f92-8f45-67631dfe95cf",
    "history_item_id": "367718752",
    "received_utc": "2026-09-10T17:44:30.379765Z",
}


def record(data: bytes) -> dict[str, object]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease", required=True)
    args = parser.parse_args()
    root = lease.lease_root(args.lease)
    intent_path = root / "SETUP_DEADLINE_AMENDMENT_INTENT.json"
    receipt_path = root / "SETUP_DEADLINE_AMENDMENT.json"

    with lease.locked(root, ".state.lock"):
        if intent_path.exists() or receipt_path.exists():
            raise ValueError("deadline amendment is single-use")
        lease_path = root / "LEASE.json"
        before_bytes = io._no_links(lease_path).read_bytes()
        before = json.loads(before_bytes)
        for key, value in EXPECTED.items():
            if before.get(key) != value:
                raise ValueError(f"unexpected pre-amendment {key}")
        if any(root.rglob("STOP")) or any(root.rglob("HALT")):
            raise ValueError("lease has a stop marker")
        forbidden = ("worker_config.json", "BOOTSTRAP_STARTED.json", "BANKS_READY.json")
        if any(any(root.rglob(name)) for name in forbidden):
            raise ValueError("worker launch state already exists")
        if datetime.now(timezone.utc).timestamp() >= lease.epoch(EXPECTED["setup_deadline_utc"]):
            raise ValueError("original setup deadline has already passed")
        if lease.epoch(NEW_SETUP_DEADLINE) >= lease.epoch(EXPECTED["work_deadline_utc"]):
            raise ValueError("amended setup deadline does not precede work deadline")

        intent = {
            "schema": "confirmation_setup_deadline_amendment_intent.v1",
            "recorded_utc": lease.stamp(),
            "authorization": AUTHORIZATION,
            "reason": "User requested more setup time after resumable bank transfer delays.",
            "scope": {"field": "setup_deadline_utc", "before": EXPECTED["setup_deadline_utc"],
                      "after": NEW_SETUP_DEADLINE},
            "unchanged_hard_limits": {
                "work_deadline_utc": EXPECTED["work_deadline_utc"],
                "watch_deadline_utc": EXPECTED["watch_deadline_utc"],
                "provider_deadline_utc": EXPECTED["provider_deadline_utc"],
                "job_incremental_cap_usd": EXPECTED["job_incremental_cap_usd"],
                "combined_cap_usd": EXPECTED["combined_cap_usd"],
            },
            "original_lease_record": record(before_bytes),
            "original_lease_base64": base64.b64encode(before_bytes).decode("ascii"),
        }
        io._atomic_json(intent_path, intent)

        after = json.loads(json.dumps(before))
        after["setup_deadline_utc"] = NEW_SETUP_DEADLINE
        io._atomic_json(lease_path, after)
        after_bytes = io._no_links(lease_path).read_bytes()
        observed = json.loads(after_bytes)
        changed = {key for key in set(before) | set(observed) if before.get(key) != observed.get(key)}
        if changed != {"setup_deadline_utc"} or observed["setup_deadline_utc"] != NEW_SETUP_DEADLINE:
            raise RuntimeError("amendment changed fields outside its authorization")
        receipt = {
            "schema": "confirmation_setup_deadline_amendment.v1",
            "recorded_utc": lease.stamp(),
            "authorization": AUTHORIZATION,
            "changed_fields": sorted(changed),
            "before": record(before_bytes),
            "after": record(after_bytes),
            "intent": {"path": str(intent_path), **record(intent_path.read_bytes())},
            "unchanged_hard_limits": intent["unchanged_hard_limits"],
        }
        io._atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
