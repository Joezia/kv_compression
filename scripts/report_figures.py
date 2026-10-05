#!/usr/bin/env python3
"""Cross-model stability figure for the v2 report.

Panel A: every turn's BF16 K/V compression ratio (primary codec), one column
per session, grouped by model. Panel B: Key vs Value ratio per model and K/V
format. Data are recomputed from each run's turn_codec.csv / summary tables.

Usage: report_figures.py --run-id <new_id> --summary <n4b_summary_run_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Chinese labels: use the first installed CJK font, keep DejaVu as fallback.
_CJK = [f for f in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "PingFang SC")
        if any(f == x.name for x in font_manager.fontManager.ttflist)]
plt.rcParams["font.sans-serif"] = _CJK + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"qwen35_9b": "#2a78d6", "qwen3_4b_2507": "#eb6834"}
NAMES = {"qwen35_9b": "Qwen3.5-9B", "qwen3_4b_2507": "Qwen3-4B"}
TEXT, MUTED, GRID = "#52514e", "#8a8985", "#e6e5e0"
PRIMARY = ("ans_lane_shared", "64")


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=TEXT, labelsize=8)


def session_label(manifest: dict) -> str:
    meta = manifest["session_meta"]
    if manifest["source_kind"].startswith("AgentX"):
        return f"AgentX t{meta['trace_index']}"
    return f"SWE {meta['session_id'].split('#')[0].split('__')[-1][:14]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    summary_dir = ROOT / "results" / args.summary
    runs = json.loads((summary_dir / "evidence.json").read_text())["source_runs"]

    rows = []
    for run_id in runs:
        d = ROOT / "results" / run_id
        manifest = json.loads((d / "manifest.json").read_text())
        per: dict[int, list[int]] = {}
        with (d / "turn_codec.csv").open() as f:
            for r in csv.DictReader(f):
                if (r["format"] == "bf16" and r["state_kind"] in ("key", "value")
                        and (r["codec"], r["block_tokens"]) == PRIMARY):
                    a = per.setdefault(int(r["turn_idx"]), [0, 0])
                    a[0] += int(r["raw_bytes"])
                    a[1] += int(r["compressed_total_bytes"])
        for turn, (raw, comp) in sorted(per.items()):
            rows.append({"run_id": run_id, "model": manifest["model"]["key"], "session": session_label(manifest),
                         "source": "AgentX" if manifest["source_kind"].startswith("AgentX") else "SWE-agent",
                         "turn_idx": turn, "raw_bytes": raw, "compressed_total_bytes": comp, "ratio": raw / comp})
    with (out / "per_turn_kv_ratio_all_sessions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    codec = list(csv.DictReader((summary_dir / "codec_table.csv").open()))

    def kv(model, fmt, scope):
        for r in codec:
            if (r["model"], r["format"], r["scope"], r["codec"], r["block_tokens"]) == (model, fmt, scope, *PRIMARY):
                return float(r["compression_ratio"])
        raise KeyError((model, fmt, scope))

    models = ["qwen35_9b", "qwen3_4b_2507"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [3, 2]})

    # Panel A: per-turn strip per session, grouped by model.
    rng = random.Random(0)
    x = 0
    ticks, labels, stats = [], [], {}
    for m in models:
        sessions = []
        for r in rows:
            if r["model"] == m and r["run_id"] not in sessions:
                sessions.append(r["run_id"])
        label_of = {r["run_id"]: r["session"] for r in rows}
        sessions.sort(key=lambda rid: (0 if label_of[rid].startswith("AgentX") else 1, label_of[rid]))
        start = x
        for rid in sessions:
            pts = [r for r in rows if r["run_id"] == rid]
            xs = [x + rng.uniform(-0.22, 0.22) for _ in pts]
            ax.scatter(xs, [p["ratio"] for p in pts], s=10, color=COLORS[m], alpha=0.75, linewidths=0)
            ticks.append(x)
            labels.append(pts[0]["session"])
            x += 1
        vals = [r["ratio"] for r in rows if r["model"] == m]
        stats[m] = (min(vals), max(vals), len(vals))
        ax.hlines([min(vals), max(vals)], start - 0.4, x - 0.6, colors=COLORS[m], linewidth=1, linestyles="dashed")
        ax.text((start + x - 1) / 2, max(vals) + 0.0006, f"{NAMES[m]}：{len(vals)} 轮，{min(vals):.4f}–{max(vals):.4f}",
                ha="center", fontsize=9, color=TEXT)
        x += 1  # gap between models
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("新增 K/V 压缩比（BF16，共享表，64 token）", color=TEXT, fontsize=9)
    ax.set_title("A. 每个点是一轮：同一模型内跨会话、跨轮次、跨内容几乎不变，两个模型之间完全不重叠",
                 fontsize=10, color=TEXT, loc="left")
    lo = min(s[0] for s in stats.values())
    hi = max(s[1] for s in stats.values())
    ax.set_ylim(lo - 0.0015, hi + 0.0022)
    style(ax)

    # Panel B: Key vs Value per model and format.
    groups = [("bf16", "BF16"), ("fp8_e4m3_sat", "FP8-E4M3")]
    width = 0.18
    for gi, (fmt, fname) in enumerate(groups):
        for mi, m in enumerate(models):
            for ki, (scope, sname) in enumerate((("key", "K"), ("value", "V"))):
                v = kv(m, fmt, scope)
                pos = gi * 1.2 + mi * 0.45 + ki * width
                bx.bar(pos, v, width * 0.92, color=COLORS[m], alpha=1.0 if scope == "key" else 0.55)
                bx.text(pos, v + 0.006, f"{sname}\n{v:.3f}", ha="center", fontsize=7.5, color=TEXT)
    bx.set_xticks([gi * 1.2 + 0.32 for gi in range(len(groups))])
    bx.set_xticklabels([f"{g[1]}（左：Qwen3.5-9B，右：Qwen3-4B）" for g in groups], fontsize=8.5)
    bx.set_ylim(1.0, 1.56)
    bx.set_ylabel("压缩比", color=TEXT, fontsize=9)
    bx.set_title("B. K 与 V 谁更好压因模型而异（深色 K，浅色 V）：Qwen3.5 是 K > V，Qwen3-4B 是 V > K",
                 fontsize=10, color=TEXT, loc="left")
    style(bx)

    fig.tight_layout()
    fig.savefig(out / "stability_by_model.png", dpi=170)
    (out / "evidence.json").write_text(json.dumps({
        "source_summary": args.summary, "source_runs": runs,
        "per_model_turn_ratio_min_max_n": {m: list(s) for m, s in stats.items()},
        "models_overlap": not (stats["qwen3_4b_2507"][1] < stats["qwen35_9b"][0] or stats["qwen35_9b"][1] < stats["qwen3_4b_2507"][0]),
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
