#!/usr/bin/env python3
"""Recompute every number quoted in the v2 report from committed result files.

Usage: report_tables.py --run-id <new_id> --summary <n4b_summary_run_id> --trace-structure <n0_run_id>
Writes storage_by_session.csv, per_turn_stats.csv, trace_structure_stats.json and
headline.json into results/<new_id>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GiB = 2**30


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def q(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--trace-structure", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    summary_dir = ROOT / "results" / args.summary
    runs = json.loads((summary_dir / "evidence.json").read_text())["source_runs"]

    storage_rows, turn_rows = [], []
    for run_id in runs:
        d = ROOT / "results" / run_id
        m = json.loads((d / "manifest.json").read_text())
        s = json.loads((d / "storage.json").read_text())
        v = json.loads((d / "verification.json").read_text())
        env = json.loads((d / "env_summary.json").read_text())
        turns = read_csv(d / "turns.csv")
        B = m["options"]["cache_block_tokens"]
        stored_c = s.get("stored_compressed_bytes_kv_plus_states_plus_tables")
        storage_rows.append({
            "run_id": run_id,
            "model": m["model"]["key"],
            "source": "AgentX" if m["source_kind"].startswith("AgentX") else "SWE-agent trajectory",
            "session_id": m["session_id"],
            "turns_processed": v["turns_processed"],
            "turns_available": v["turns_available"],
            "complete_session": v["truncated_reason"] is None,
            "final_context_tokens": int(turns[-1]["full_blocks"]) * B,
            "rollbacks": sum(t["restore"] != "none" for t in turns),
            "snapshot_kv_GiB": s["snapshot_kv_bytes"] / GiB,
            "unique_kv_GiB": s["unique_kv_bytes"] / GiB,
            "state_snapshot_GiB": s["state_snapshot_bytes"] / GiB,
            "stored_compressed_GiB": stored_c / GiB if stored_c else None,
            "R_dedup_kv": s["R_dedup_kv"],
            "R_codec_kv": s.get("R_codec_kv"),
            "R_codec_stored": s.get("R_codec_stored"),
            "state_share_of_stored_raw": s["state_snapshot_bytes"] / (s["unique_kv_bytes"] + s["state_snapshot_bytes"]),
            "R_end_to_end_vs_snapshot_kv_only": (s["snapshot_kv_bytes"] / s["compressed_unique_kv_bytes"]) if s.get("compressed_unique_kv_bytes") else None,
            "bit_exact_units": v["bit_exact_units"],
            "rehydration": (v.get("rehydration") or {}).get("status"),
            "cuda_visible_devices": env.get("cuda_visible_devices"),
            "wall_seconds": v["wall_seconds"],
            "gpu_peak_reserved_GiB": max(int(t["gpu_peak_reserved_bytes"]) for t in turns) / GiB,
        })
        incr = [t for t in turns[2:]] if m["source_kind"].startswith("AgentX") else turns[1:]
        new_tok = [int(t["new_blocks"]) * B for t in incr]
        kv_b = [int(t["new_kv_raw_bytes"]) for t in incr]
        st_b = [int(t["state_snapshot_raw_bytes"]) for t in incr]
        turn_rows.append({
            "run_id": run_id,
            "model": m["model"]["key"],
            "source": storage_rows[-1]["source"],
            "incremental_turns": len(incr),
            "first_large_turn_tokens": int(turns[1]["new_blocks"]) * B if m["source_kind"].startswith("AgentX") else int(turns[0]["new_blocks"]) * B,
            "median_new_tokens_per_turn": st.median(new_tok) if new_tok else None,
            "p90_new_tokens_per_turn": q(new_tok, 0.9) if new_tok else None,
            "median_new_kv_MiB_per_turn": st.median(kv_b) / 2**20 if kv_b else None,
            "median_state_MiB_per_turn": st.median(st_b) / 2**20 if st_b else None,
            "turns_where_state_exceeds_new_kv": sum(1 for a, b in zip(kv_b, st_b) if b > a),
            "median_forward_seconds": st.median(float(t["forward_seconds"]) for t in incr) if incr else None,
        })
    with (out / "storage_by_session.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(storage_rows[0]))
        w.writeheader()
        w.writerows(storage_rows)
    with (out / "per_turn_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(turn_rows[0]))
        w.writeheader()
        w.writerows(turn_rows)

    # AgentX trace structure (N0): hash-level, no model involved.
    n0 = ROOT / "results" / args.trace_structure
    sess = read_csv(n0 / "trace_sessions.csv")
    tt = read_csv(n0 / "trace_turns.csv")
    incr_new = [int(t["new_blocks"]) * 64 + int(t["uncovered_tail_tokens"]) for t in tt if int(t["turn_idx"]) > 0]
    hit_frac = [int(t["hit_blocks"]) / int(t["full_blocks"]) for t in tt if int(t["turn_idx"]) > 0 and int(t["full_blocks"])]
    gaps = [float(t["gap_s"]) for t in tt if t["gap_s"] not in ("", "None")]
    trace = {
        "sessions": len(sess),
        "main_turns": len(tt),
        "overall_dedup_ratio": json.loads((n0 / "summary.json").read_text())["dedup_ratio_full_blocks"],
        "per_session_dedup_p10_p50_p90": [q([float(s["dedup_ratio_tokens"]) for s in sess], p) for p in (0.1, 0.5, 0.9)],
        "main_turns_p10_p50_p90_max": [q([int(s["main_turns"]) for s in sess], p) for p in (0.1, 0.5, 0.9)] + [max(int(s["main_turns"]) for s in sess)],
        "max_input_tokens_p10_p50_p90_max": [q([int(s["max_input_tokens"]) for s in sess], p) for p in (0.1, 0.5, 0.9)] + [max(int(s["max_input_tokens"]) for s in sess)],
        "new_tokens_per_turn_p50_p90": [q(incr_new, 0.5), q(incr_new, 0.9)],
        "prefix_hit_fraction_per_turn_p10_p50": [q(hit_frac, 0.1), q(hit_frac, 0.5)],
        "inter_request_gap_s_p50_p90": [q(gaps, 0.5), q(gaps, 0.9)] if gaps else None,
        "sessions_with_subagents": sum(int(s["subagent_requests"]) > 0 for s in sess),
    }
    (out / "trace_structure_stats.json").write_text(json.dumps(trace, indent=2) + "\n")

    codec = read_csv(summary_dir / "codec_table.csv")

    def ratio(model, fmt, scope, codec_name, block):
        for r in codec:
            if (r["model"], r["format"], r["scope"], r["codec"], int(r["block_tokens"])) == (model, fmt, scope, codec_name, block):
                return {k: float(r[k]) for k in ("compression_ratio", "payload_only_ratio", "order0_entropy_ratio",
                                                 "session_min_ratio", "session_max_ratio")}
        return None

    headline = {
        "all_units_bit_exact": json.loads((summary_dir / "evidence.json").read_text())["all_units_bit_exact"],
        "bit_exact_units": json.loads((summary_dir / "evidence.json").read_text())["bit_exact_units"],
        "kv_bf16_primary": {m: ratio(m, "bf16", "kv_only", "ans_lane_shared", 64) for m in ("qwen35_9b", "qwen3_4b_2507")},
        "kv_bf16_zstd": {m: ratio(m, "bf16", "kv_only", "zstd", 64) for m in ("qwen35_9b", "qwen3_4b_2507")},
        "kv_fp8_primary": {m: ratio(m, "fp8_e4m3_sat", "kv_only", "ans_lane_shared", 64) for m in ("qwen35_9b", "qwen3_4b_2507")},
        "qwen35_state_snapshots_primary": ratio("qwen35_9b", "native", "state_snapshots", "ans_lane_shared", -1),
        "by_source_R_codec_kv": {
            f"{r['model']}|{r['source']}": round(st.mean(float(x["R_codec_kv"]) for x in storage_rows
                                                         if x["model"] == r["model"] and x["source"] == r["source"]), 4)
            for r in storage_rows
        },
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
