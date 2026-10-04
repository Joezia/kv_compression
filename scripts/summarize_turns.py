#!/usr/bin/env python3
"""Aggregate per-turn runs (run_turns.py run) into tables and figures.

Usage: summarize_turns.py --run-id <new_summary_id> RUN_ID [RUN_ID ...]
All numbers are recomputed from each run's CSV/JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Reference categorical order (blue, orange, aqua) + muted ink; validated set.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#8a8985"
TEXT = "#52514e"


def style(ax):
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=TEXT, labelsize=8)


def model_key(manifest: dict) -> str:
    return manifest["model"].get("key", manifest["model"]["id"])


def load_run(run_id: str) -> dict:
    d = ROOT / "results" / run_id
    manifest = json.loads((d / "manifest.json").read_text())
    return {
        "run_id": run_id,
        "dir": d,
        "manifest": manifest,
        "label": f"{model_key(manifest)} | {manifest['source_kind']} | {manifest['session_id'][:8]}",
        "turns": pd.read_csv(d / "turns.csv"),
        "turn_codec": pd.read_csv(d / "turn_codec.csv"),
        "summary": pd.read_csv(d / "summary.csv"),
        "storage": json.loads((d / "storage.json").read_text()),
        "verification": json.loads((d / "verification.json").read_text()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    out = ROOT / "results" / args.run_id
    (out / "figures").mkdir(parents=True, exist_ok=False)
    runs = [load_run(r) for r in args.runs]

    # ---- per-session table ------------------------------------------------
    rows = []
    for r in runs:
        s, v, m = r["storage"], r["verification"], r["manifest"]
        rows.append({
            "run_id": r["run_id"], "model": model_key(m), "source_kind": m["source_kind"],
            "session_id": m["session_id"], "turns_processed": v["turns_processed"],
            "turns_available": v["turns_available"], "truncated_reason": v["truncated_reason"],
            "final_context_tokens": int(r["turns"]["full_blocks"].iloc[-1] * m["options"]["cache_block_tokens"]),
            "rollbacks": int((r["turns"]["restore"] != "none").sum()),
            "R_dedup_kv": s.get("R_dedup_kv"), "primary": s.get("primary"), "R_codec_kv": s.get("R_codec_kv"),
            "unique_kv_bytes": s.get("unique_kv_bytes"), "state_snapshot_bytes": s.get("state_snapshot_bytes"),
            "state_share_of_stored": (s["state_snapshot_bytes"] / (s["unique_kv_bytes"] + s["state_snapshot_bytes"]))
            if s.get("unique_kv_bytes") else None,
            "R_codec_stored": s.get("R_codec_stored"),
            "bit_exact_units": v["bit_exact_units"],
            "rehydration": (v.get("rehydration") or {}).get("status"),
        })
    sessions = pd.DataFrame(rows)
    sessions.to_csv(out / "sessions.csv", index=False)

    # ---- codec table: bytes-weighted across sessions, plus per-session spread ----
    frames = []
    for r in runs:
        df = r["summary"].copy()
        df["model"] = model_key(r["manifest"])
        df["run_id"] = r["run_id"]
        frames.append(df)
    allsum = pd.concat(frames, ignore_index=True)
    allsum["block_tokens"] = allsum["block_tokens"].fillna(-1).astype(int)
    keys = ["model", "format", "scope", "codec", "block_tokens"]
    sums = allsum.groupby(keys, as_index=False)[["raw_bytes", "payload_bytes", "table_bytes", "header_bytes",
                                                 "compressed_total_bytes", "entropy_lane_bytes"]].sum()
    sums["compression_ratio"] = sums["raw_bytes"] / sums["compressed_total_bytes"]
    sums["payload_only_ratio"] = sums["raw_bytes"] / sums["payload_bytes"]
    sums["order0_entropy_ratio"] = sums["raw_bytes"] / sums["entropy_lane_bytes"]
    spread = allsum.groupby(keys)["compression_ratio"].agg(["min", "max", "count"]).reset_index()
    spread = spread.rename(columns={"min": "session_min_ratio", "max": "session_max_ratio", "count": "sessions"})
    codec_table = sums.merge(spread, on=keys)
    codec_table.to_csv(out / "codec_table.csv", index=False)

    # ---- fig 1: per-turn new tokens, cumulative K/V bytes, cumulative state bytes ----
    for r in runs:
        t = r["turns"]
        tc = r["turn_codec"]
        B = r["manifest"]["options"]["cache_block_tokens"]
        prim = r["storage"].get("primary")
        has_states = bool(t["state_snapshot_raw_bytes"].sum())
        fig, axes = plt.subplots(1, 3 if has_states else 2, figsize=(15 if has_states else 10.5, 3.8))
        ax = axes[0]
        ax.bar(t["turn_idx"], t["new_blocks"] * B, color=SERIES[0], width=0.8)
        ax.set_xlabel("turn", color=TEXT)
        ax.set_ylabel("new tokens stored (full blocks)", color=TEXT)
        ax.set_title("New K/V per turn", fontsize=10, color=TEXT)
        style(ax)

        def compressed_per_turn(kinds, block):
            if not prim:
                return None
            codec = prim.split("@")[0]
            sel = tc[(tc["codec"] == codec) & tc["state_kind"].isin(kinds)]
            sel = sel[sel["block_tokens"].isna()] if block is None else sel[(sel["block_tokens"] == float(block)) & (sel["format"] == "bf16")]
            return sel.groupby("turn_idx")["compressed_total_bytes"].sum().reindex(t["turn_idx"], fill_value=0)

        ax = axes[1]
        bpt = r["storage"].get("kv_bytes_per_token") or 0
        ax.plot(t["turn_idx"], (t["full_blocks"] * B * bpt).cumsum() / 2**30, color=MUTED, linewidth=2,
                label="full snapshot every turn")
        ax.plot(t["turn_idx"], t["new_kv_raw_bytes"].cumsum() / 2**30, color=SERIES[0], linewidth=2,
                label="prefix-dedup (new blocks only)")
        ckv = compressed_per_turn(["key", "value"], prim.split("@")[1] if prim else None)
        if ckv is not None:
            ax.plot(t["turn_idx"], ckv.cumsum().values / 2**30, color=SERIES[1], linewidth=2, label=f"dedup + {prim}")
        ax.set_xlabel("turn", color=TEXT)
        ax.set_ylabel("cumulative GiB", color=TEXT)
        ax.set_title("K/V bytes (BF16)", fontsize=10, color=TEXT)
        ax.legend(fontsize=8, frameon=False)
        style(ax)
        if has_states:
            ax = axes[2]
            ax.plot(t["turn_idx"], t["state_snapshot_raw_bytes"].cumsum() / 2**30, color=SERIES[0], linewidth=2,
                    label="state snapshots, raw")
            cst = compressed_per_turn(["recurrent", "convolution"], None)
            if cst is not None:
                ax.plot(t["turn_idx"], cst.cumsum().values / 2**30, color=SERIES[1], linewidth=2,
                        label=f"state snapshots, {prim.split('@')[0]}")
            ax.set_xlabel("turn", color=TEXT)
            ax.set_ylabel("cumulative GiB", color=TEXT)
            ax.set_title("Linear-attention state snapshots (one per turn)", fontsize=10, color=TEXT)
            ax.legend(fontsize=8, frameon=False)
            style(ax)
        fig.suptitle(r["label"], fontsize=10, color=TEXT)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"turns_{r['run_id']}.png", dpi=160)
        plt.close(fig)

    # ---- fig 2: per-turn K/V codec ratio (is it flat across turns?) ----------
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, r in enumerate(runs[:3]):
        tc = r["turn_codec"]
        sel = tc[(tc["format"] == "bf16") & tc["state_kind"].isin(["key", "value"])]
        prim = r["storage"].get("primary", "ans_lane_shared@64").split("@")
        p = sel[(sel["codec"] == prim[0]) & (sel["block_tokens"] == float(prim[1]))].groupby("turn_idx")[
            ["raw_bytes", "compressed_total_bytes"]].sum()
        ax.plot(p.index, p["raw_bytes"] / p["compressed_total_bytes"], color=SERIES[i], linewidth=2, marker="o",
                markersize=3, label=r["label"])
    ax.set_xlabel("turn", color=TEXT)
    ax.set_ylabel("raw / compressed (new K/V, BF16)", color=TEXT)
    ax.set_title("Per-turn K/V compression ratio, primary codec", fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "figures" / "per_turn_kv_ratio.png", dpi=160)
    plt.close(fig)

    # ---- fig 3: codec x block size, total vs payload-only (K/V, per model/format) ----
    kv = codec_table[codec_table["scope"] == "kv_only"]
    for (model, fmt), g in kv.groupby(["model", "format"]):
        g = g.sort_values(["codec", "block_tokens"])
        labels = [f"{c}\n{b} tok" for c, b in zip(g["codec"], g["block_tokens"])]
        x = range(len(g))
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(g)), 3.8))
        ax.bar([i - 0.2 for i in x], g["compression_ratio"], width=0.38, color=SERIES[0], label="total (incl. tables/headers)")
        ax.bar([i + 0.2 for i in x], g["payload_only_ratio"], width=0.38, color=SERIES[2], label="payload only")
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("raw / compressed", color=TEXT)
        ax.set_title(f"{model}, K/V {fmt}: codec and unit size (all sessions)", fontsize=10, color=TEXT)
        ax.legend(fontsize=8, frameon=False)
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"codec_block_{str(model).replace('/', '_')}_{fmt}.png", dpi=160)
        plt.close(fig)

    (out / "evidence.json").write_text(json.dumps({
        "source_runs": args.runs,
        "all_units_bit_exact": bool(all(r["verification"]["all_units_bit_exact"] for r in runs)),
        "bit_exact_units": int(sum(r["verification"]["bit_exact_units"] for r in runs)),
        "rehydration": {r["run_id"]: (r["verification"].get("rehydration") or {}).get("status") for r in runs},
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
