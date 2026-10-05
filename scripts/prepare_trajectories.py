#!/usr/bin/env python3
"""Export real-content agentic trajectories as normalized message lists (JSONL).

The dataset schema MUST be checked on the server first (`--inspect`); the
field names below are auto-detected candidates, not verified facts.
Selection uses only structure (turn count), never compression results.

Example:
  prepare_trajectories.py --dataset nebius/SWE-agent-trajectories --inspect
  prepare_trajectories.py --dataset nebius/SWE-agent-trajectories --revision <sha> \
      --out data/raw/swe_agent_trajectories.jsonl --count 2 --min-assistant-turns 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_inputs import normalize_messages  # noqa: E402

MESSAGE_FIELDS = ["messages", "trajectory", "conversation", "conversations", "history"]
ID_FIELDS = ["instance_id", "id", "trajectory_id", "session_id"]


def find_messages(row: dict) -> list[dict] | None:
    for name in MESSAGE_FIELDS:
        value = row.get(name)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--config-name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--min-assistant-turns", type=int, default=15)
    parser.add_argument("--distinct-field", help="skip rows whose value in this field was already selected (e.g. instance_id)")
    parser.add_argument("--inspect", action="store_true", help="print the schema of the first row and exit")
    args = parser.parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config_name, split=args.split, revision=args.revision, streaming=True)
    if args.inspect:
        row = next(iter(ds))
        print(json.dumps({k: (type(v).__name__, str(v)[:300]) for k, v in row.items()}, indent=2))
        raw = find_messages(row)
        if raw is not None:
            print("message keys:", sorted({k for m in raw for k in m}))
            print("roles:", dict(Counter(str(m.get("role", m.get("from"))) for m in raw)))
            msgs = normalize_messages(raw)
            print("normalized:", [(m["role"], len(m["content"])) for m in msgs[:6]], "...")
            print("empty contents after normalization:", sum(not m["content"] for m in msgs))
        hard_exit()
    if args.out is None:
        parser.error("--out is required")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen_values: set[str] = set()
    with args.out.open("x", encoding="utf-8") as f:
        for index, row in enumerate(ds):
            raw = find_messages(row)
            if raw is None:
                continue
            msgs = normalize_messages(raw)
            if any(not m["content"] for m in msgs if m["role"] == "system"):
                raise SystemExit(f"row {index}: empty system message after normalization; check the schema")
            n_assistant = sum(m["role"] == "assistant" for m in msgs)
            if n_assistant < args.min_assistant_turns:
                continue
            if args.distinct_field:
                value = str(row.get(args.distinct_field))
                if value in seen_values:
                    continue
                seen_values.add(value)
            sid = next((str(row[k]) for k in ID_FIELDS if row.get(k) is not None), f"row{index}")
            f.write(json.dumps({
                "session_id": f"{sid}#{index}", "row_index": index, "dataset": args.dataset,
                "revision": args.revision, "source_kind": f"trajectory:{args.dataset}",
                "assistant_turns": n_assistant, "messages": msgs,
                "selection_rule": f"first {args.count} rows with >= {args.min_assistant_turns} assistant turns"
                + (f", distinct {args.distinct_field}" if args.distinct_field else ""),
            }) + "\n")
            kept += 1
            if kept >= args.count:
                break
    print(args.out, kept)
    hard_exit()


def hard_exit() -> None:
    """Skip interpreter finalization: datasets' streaming threads can abort it
    (PyGILState_Release fatal error) after all output is already written."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
