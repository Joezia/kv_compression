#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "20261004_m2a_qwen35_8k_allcodecs",
    "20261004_m2b_qwen35_trace1_2k_ansraw",
    "20261004_m2c_qwen35_trace2_2k_ansraw",
]
OUT = ROOT / "results" / "20261004_m3_summary"
FIGURES = OUT / "figures"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    FIGURES.mkdir(parents=True)
    frames = []
    inventory = []
    for run_id in RUNS:
        run_dir = ROOT / "results" / run_id
        frame = pd.read_csv(run_dir / "blocks.csv")
        frame["source_run_id"] = run_id
        if not frame["bit_exact"].astype(bool).all():
            raise AssertionError(f"non-bit-exact block in {run_id}")
        frames.append(frame)
        audit = read_json(run_dir / "cache_audit.json")
        manifest = json.loads((run_dir / "manifest.jsonl").read_text(encoding="utf-8"))
        verification = read_json(run_dir / "verification.json")
        inventory.append(
            {
                "run_id": run_id,
                "trace_id": manifest["trace_id"],
                "request_path": manifest["request_path"],
                "input_tokens": audit["input_tokens"],
                "kv_only_raw_bytes": audit["kv_only_raw_bytes"],
                "complete_persistent_state_raw_bytes": audit[
                    "complete_persistent_state_raw_bytes"
                ],
                "forward_seconds": audit["forward_seconds"],
                "cuda_peak_allocated_bytes": audit["cuda_peak_allocated_bytes"],
                "all_blocks_bit_exact": verification["all_blocks_bit_exact"],
                "rehydrated_cache_byte_exact": verification[
                    "rehydrated_cache_byte_exact"
                ],
                "continuation_hidden_exact": verification[
                    "continuation_hidden_exact"
                ],
            }
        )
    all_blocks = pd.concat(frames, ignore_index=True)
    pd.DataFrame(inventory).to_csv(OUT / "run_inventory.csv", index=False)

    primary = all_blocks[
        (all_blocks["codec"] == "ans_raw")
        & (all_blocks["block_size"] == 262144)
    ]
    aggregate_rows = []
    for scope, selector in [
        ("kv_only", primary["state_kind"].isin(["key", "value"])),
        ("complete_persistent_state", pd.Series(True, index=primary.index)),
        ("key", primary["state_kind"] == "key"),
        ("value", primary["state_kind"] == "value"),
        ("recurrent", primary["state_kind"] == "recurrent"),
        ("convolution", primary["state_kind"] == "convolution"),
    ]:
        selected = primary[selector]
        raw = int(selected["raw_bytes"].sum())
        compressed = int(selected["compressed_total_bytes"].sum())
        aggregate_rows.append(
            {
                "codec": "ans_raw",
                "block_size": 262144,
                "scope": scope,
                "requests": len(RUNS),
                "sessions": len({row["trace_id"] for row in inventory}),
                "blocks": len(selected),
                "raw_bytes": raw,
                "compressed_total_bytes": compressed,
                "compression_ratio": raw / compressed,
                "space_saving": 1 - compressed / raw,
                "bit_exact_blocks": int(selected["bit_exact"].astype(bool).sum()),
            }
        )
    with (OUT / "aggregate_raw_ans_256k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    main_run = frames[0]
    grouped = (
        main_run.groupby(["codec", "block_size", "state_kind"], as_index=False)[
            ["raw_bytes", "compressed_total_bytes"]
        ]
        .sum()
    )
    grouped["compression_ratio"] = grouped["raw_bytes"] / grouped[
        "compressed_total_bytes"
    ]
    grouped.to_csv(OUT / "m2a_state_kind_summary.csv", index=False)

    scope_rows = []
    for (codec, block_size), group in main_run.groupby(["codec", "block_size"]):
        for scope, selected in [
            ("KV-only", group[group["state_kind"].isin(["key", "value"])]),
            ("Complete state", group),
        ]:
            raw = selected["raw_bytes"].sum()
            compressed = selected["compressed_total_bytes"].sum()
            scope_rows.append(
                {
                    "codec": codec,
                    "block_size": block_size,
                    "scope": scope,
                    "compression_ratio": raw / compressed,
                }
            )
    scope_df = pd.DataFrame(scope_rows)
    scope_df.to_csv(OUT / "m2a_codec_scope_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, scope in zip(axes, ["KV-only", "Complete state"], strict=True):
        subset = scope_df[
            (scope_df["scope"] == scope) & (scope_df["block_size"] == 262144)
        ]
        axis.bar(subset["codec"], subset["compression_ratio"])
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(scope)
        axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("raw / compressed_total")
    fig.suptitle("Qwen3.5-9B, AgentX-prefix-sampled, 8K, 256 KiB")
    fig.tight_layout()
    fig.savefig(FIGURES / "codec_scope_ratio.png", dpi=180)
    plt.close(fig)

    layer = main_run[
        (main_run["codec"] == "ans_byte_lane")
        & (main_run["block_size"] == 262144)
    ]
    layer = layer.groupby(["layer_idx", "state_kind"], as_index=False)[
        ["raw_bytes", "compressed_total_bytes"]
    ].sum()
    layer["compression_ratio"] = layer["raw_bytes"] / layer[
        "compressed_total_bytes"
    ]
    layer.to_csv(OUT / "m2a_layer_state_summary.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for kind, values in layer.groupby("state_kind"):
        axis.plot(values["layer_idx"], values["compression_ratio"], marker="o", label=kind)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xlabel("layer index")
    axis.set_ylabel("raw / compressed_total")
    axis.set_title("Byte-lane ANS by persistent state, 8K, 256 KiB")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "layer_state_ratio.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, scope in zip(axes, ["KV-only", "Complete state"], strict=True):
        subset = scope_df[scope_df["scope"] == scope]
        for codec, values in subset.groupby("codec"):
            values = values.sort_values("block_size")
            axis.plot(
                values["block_size"] / 1024,
                values["compression_ratio"],
                marker="o",
                label=codec,
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(scope)
        axis.set_xlabel("block size (KiB)")
    axes[0].set_ylabel("raw / compressed_total")
    axes[1].legend()
    fig.suptitle("Block-size effect, Qwen3.5-9B, AgentX-prefix-sampled, 8K")
    fig.tight_layout()
    fig.savefig(FIGURES / "block_size_ratio.png", dpi=180)
    plt.close(fig)

    evidence = {
        "source_runs": RUNS,
        "all_blocks": len(all_blocks),
        "all_blocks_bit_exact": bool(all_blocks["bit_exact"].astype(bool).all()),
        "note": "M1 is excluded from aggregate because it is the 2K prefix of the M2a request.",
    }
    (OUT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
