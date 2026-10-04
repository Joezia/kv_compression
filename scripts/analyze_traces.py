#!/usr/bin/env python3
"""CPU-only AgentX session structure analysis (no tokenizer, no GPU).

Per main-thread turn: input length, prefix hit vs. new 64-token blocks (prefix
chain over hash_ids), timing. Per session: snapshot vs. unique tokens, i.e. the
dedup ratio that prefix caching gives before any entropy coding. Bytes are
derived analytically from per-token K/V size and per-turn hybrid state size.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from agentic_inputs import iter_trace_rows, load_trace_rows, trace_turn_stats  # noqa: E402

# Per-token K/V bytes and per-snapshot state bytes. Qwen3.5-9B values are the
# M0-audited runtime numbers; others must be confirmed by `run_turns.py audit`.
MODEL_BYTES = {
    "qwen35_9b": {"kv_bytes_per_token": 32768, "state_bytes_per_snapshot": 50331648 + 1572864, "audited": True},
    "qwen3_4b_2507": {"kv_bytes_per_token": 36 * 2 * 8 * 128 * 2, "state_bytes_per_snapshot": 0, "audited": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "turns.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=50, help="analyze the first N traces of the dataset")
    parser.add_argument("--cached-only", action="store_true", help="use only data/raw/agentx_trace_*.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["dataset"]
    out = ROOT / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)

    if args.cached_only:
        cache_dir = ROOT / "data" / "raw"
        indices = sorted(int(p.stem.rsplit("_", 1)[1]) for p in cache_dir.glob("agentx_trace_*.json"))
        rows = load_trace_rows(data["id"], data["revision"], data["split"], indices, cache_dir).items()
    else:
        rows = iter_trace_rows(data["id"], data["revision"], data["split"], args.limit)

    turn_out, sessions = [], []
    for index, row in rows:
        turns, summary = trace_turn_stats(row)
        summary["trace_index"] = index
        for name, mb in MODEL_BYTES.items():
            b = mb["kv_bytes_per_token"]
            stored_turns = sum(1 for t in turns if t["new_blocks"] > 0)
            summary[f"{name}_snapshot_kv_bytes"] = summary["snapshot_tokens"] * b
            summary[f"{name}_unique_kv_bytes"] = summary["unique_block_tokens"] * b
            summary[f"{name}_state_snapshot_bytes"] = stored_turns * mb["state_bytes_per_snapshot"]
        sessions.append(summary)
        for t in turns:
            turn_out.append({"trace_index": index, "session_id": summary["session_id"], **t})

    with (out / "trace_turns.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(turn_out[0]))
        w.writeheader()
        w.writerows(turn_out)
    with (out / "trace_sessions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sessions[0]))
        w.writeheader()
        w.writerows(sessions)

    snap = sum(s["snapshot_tokens"] for s in sessions)
    uniq = sum(s["unique_block_tokens"] for s in sessions)
    new_incl_tail = sum(s["new_tokens_incl_tail"] for s in sessions)
    summary = {
        "dataset": f"{data['id']}@{data['revision']}",
        "sessions": len(sessions),
        "main_turns": len(turn_out),
        "snapshot_tokens": snap,
        "unique_block_tokens": uniq,
        "dedup_ratio_full_blocks": snap / uniq if uniq else None,
        "dedup_ratio_incl_uncached_tail": snap / new_incl_tail if new_incl_tail else None,
        "model_bytes": MODEL_BYTES,
        "note": "hash-level structure only; content is not involved. Subagent requests counted, not deduplicated.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
