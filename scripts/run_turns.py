#!/usr/bin/env python3
"""Per-turn agentic KV/state experiments.

  audit      short real forward: layer types, K/V and state shapes/dtypes/bytes
  calibrate  build shared ANS tables from held-out session(s)
  run        per-turn capture + coding for one session

Every invocation needs a new --run-id; nothing is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from agentic_inputs import (  # noqa: E402
    Session,
    agentx_session,
    load_trace_rows,
    session_fingerprint,
    trajectory_session,
)
from pilot import environment_summary  # noqa: E402
from turn_runner import (  # noqa: E402
    AGG_FIELDS,
    RunOptions,
    TurnRunner,
    agg_rows,
    build_tables,
    kv_tensors,
    layer_types_of,
    linear_state_tensors,
    load_model,
    load_tables,
    new_cache,
    save_tables,
    table_set_bytes,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def resolve_revision(spec: dict) -> str | None:
    if spec.get("revision"):
        return spec["revision"]
    from huggingface_hub import HfApi

    return HfApi().model_info(spec["id"]).sha


def build_session(cfg: dict, spec: dict, args, trace_index: int | None = None, max_turns: int | None = None) -> Session:
    if args.source == "agentx":
        data = cfg["dataset"]
        idx = trace_index if trace_index is not None else args.trace_index
        row = load_trace_rows(data["id"], data["revision"], data["split"], [idx], ROOT / "data" / "raw")[idx]
        session = agentx_session(
            row, tokenizer_id=spec["id"], tokenizer_revision=spec["revision"], corpus=data["corpus"],
            seed=int(cfg["seed"]), max_turns=max_turns,
        )
        session.meta.update({"trace_index": idx, "dataset": f"{data['id']}@{data['revision']}"})
        return session
    if args.source == "trajectory":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
        records = [json.loads(line) for line in Path(args.trajectory_file).read_text().splitlines() if line.strip()]
        rec = records[args.trajectory_index]
        session = trajectory_session(
            rec["session_id"], rec["messages"], tokenizer=tok, source_kind=rec.get("source_kind", "trajectory"),
            max_turns=max_turns,
        )
        session.meta.update({k: v for k, v in rec.items() if k != "messages"})
        session.meta["trajectory_file"] = args.trajectory_file
        return session
    raise ValueError(args.source)


def options_from_cfg(cfg: dict, spec: dict, mode: str, max_turns: int | None) -> RunOptions:
    run, rt = cfg["run"], cfg["runtime"]
    return RunOptions(
        mode=mode,
        cache_block_tokens=int(run["cache_block_tokens"]),
        prefill_chunk_tokens=int(run["prefill_chunk_tokens"]),
        max_context_tokens=int(spec["max_context_tokens"]),
        max_turns=max_turns,
        max_snapshots=int(run["max_snapshots"]),
        kv_formats=list(run["kv_formats"]),
        kv_codecs=[(c, int(b)) for c, b in run["kv_codecs"]],
        state_codecs=list(run["state_codecs"]),
        primary=(run["primary"][0], int(run["primary"][1])) if run.get("primary") else None,
        zstd_level=int(run["zstd_level"]),
        cpu_workers=int(rt["cpu_workers"]),
        max_inflight_bytes=int(float(rt["max_inflight_gib"]) * (1 << 30)),
        max_gpu_memory_bytes=int(float(rt["max_gpu_memory_gib"]) * (1 << 30)),
    )


SCOPES = {
    "kv_only": {"key", "value"},
    "key": {"key"},
    "value": {"value"},
    "recurrent": {"recurrent"},
    "convolution": {"convolution"},
    "state_snapshots": {"recurrent", "convolution"},
}


def summarize(turn_codec: list[dict]) -> list[dict]:
    totals: dict[tuple, list] = defaultdict(lambda: [0] * len(AGG_FIELDS))
    for r in turn_codec:
        for scope, kinds in SCOPES.items():
            if r["state_kind"] in kinds:
                key = (r["format"], scope, r["codec"], r["block_tokens"])
                for i, f in enumerate(AGG_FIELDS):
                    totals[key][i] += r[f]
    rows = []
    for (fmt, scope, codec, block), v in sorted(totals.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        d = dict(zip(AGG_FIELDS, v))
        rows.append({
            "format": fmt, "scope": scope, "codec": codec, "block_tokens": block, **d,
            "compression_ratio": d["raw_bytes"] / d["compressed_total_bytes"],
            "payload_only_ratio": d["raw_bytes"] / d["payload_bytes"] if d["payload_bytes"] else None,
            "order0_entropy_ratio": d["raw_bytes"] / d["entropy_lane_bytes"] if d["entropy_lane_bytes"] else None,
            "space_saving": 1 - d["compressed_total_bytes"] / d["raw_bytes"],
        })
    return rows


def storage_accounting(turns: list[dict], summary: list[dict], primary, B: int, table_bytes: int) -> dict:
    stored = [t for t in turns if t["new_blocks"] > 0]
    kv_per_token = None
    for t in stored:
        kv_per_token = t["new_kv_raw_bytes"] / (t["new_blocks"] * B)
        break
    snapshot_full_block_tokens = sum(t["full_blocks"] * B for t in turns)
    unique_tokens = sum(t["new_blocks"] * B for t in turns)
    unique_kv = sum(t["new_kv_raw_bytes"] for t in turns)
    states = sum(t["state_snapshot_raw_bytes"] for t in turns)

    def compressed(scope, codec, block):
        for r in summary:
            if r["format"] in ("bf16", "native") and r["scope"] == scope and r["codec"] == codec and r["block_tokens"] == block:
                return r["compressed_total_bytes"]
        return None

    out = {
        "turns_processed": len(turns),
        "turns_with_new_blocks": len(stored),
        "input_tokens_all_turns": sum(t["input_tokens"] for t in turns),
        "snapshot_full_block_tokens": snapshot_full_block_tokens,
        "unique_block_tokens": unique_tokens,
        "kv_bytes_per_token": kv_per_token,
        "snapshot_kv_bytes": snapshot_full_block_tokens * kv_per_token if kv_per_token else None,
        "unique_kv_bytes": unique_kv,
        "state_snapshot_bytes": states,
        "R_dedup_kv": snapshot_full_block_tokens / unique_tokens if unique_tokens else None,
        "shared_table_set_bytes": table_bytes,
    }
    if primary:
        codec, block = primary
        ckv = compressed("kv_only", codec, block)
        cst = compressed("state_snapshots", codec, None) if states else 0
        if ckv:
            out["primary"] = f"{codec}@{block}"
            out["compressed_unique_kv_bytes"] = ckv
            out["R_codec_kv"] = unique_kv / ckv
            out["compressed_state_bytes"] = cst
            if cst is not None:
                total_c = ckv + cst + (table_bytes if codec.endswith("shared") else 0)
                out["stored_raw_bytes_kv_plus_states"] = unique_kv + states
                out["stored_compressed_bytes_kv_plus_states_plus_tables"] = total_c
                out["R_codec_stored"] = (unique_kv + states) / total_c
    return out


def cmd_audit(cfg, spec, args, run_dir):
    device = torch.device(args.device)
    session = build_session(cfg, spec, args, max_turns=2)
    tokens = next(t.token_ids for t in session.turns if len(t.token_ids) >= args.audit_tokens)[: args.audit_tokens]
    model = load_model(spec, device)
    types = layer_types_of(model)
    cache = new_cache(model)
    with torch.inference_mode():
        model.model(input_ids=torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0).to(device),
                    past_key_values=cache, use_cache=True)
    layers = []
    for i, kind in enumerate(types):
        if kind == "full_attention":
            k, v = kv_tensors(cache, i)
            layers.append({"layer": i, "type": kind, "key": [list(k.shape), str(k.dtype)], "value": [list(v.shape), str(v.dtype)]})
        else:
            layers.append({"layer": i, "type": kind, "states": [[s, j, list(t.shape), str(t.dtype)] for s, j, t in linear_state_tensors(cache, i)]})
    kv_per_token = sum(
        kv_tensors(cache, i)[0][0, :, :1, :].numel() * kv_tensors(cache, i)[0].element_size() * 2
        for i, kind in enumerate(types) if kind == "full_attention"
    )
    state_bytes = sum(t.numel() * t.element_size() for i, kind in enumerate(types) if kind == "linear_attention"
                      for _, _, t in linear_state_tensors(cache, i))
    write_json(run_dir / "cache_audit.json", {
        "model": spec, "audit_tokens": len(tokens), "layer_type_counts": {k: types.count(k) for k in set(types)},
        "kv_bytes_per_token": kv_per_token, "state_bytes_per_snapshot": state_bytes, "layers": layers,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    })
    print(json.dumps({"kv_bytes_per_token": kv_per_token, "state_bytes_per_snapshot": state_bytes}, indent=2))


def cmd_calibrate(cfg, spec, args, run_dir):
    device = torch.device(args.device)
    model = load_model(spec, device)
    histograms: dict = {}
    sources = []
    for idx in cfg["dataset"]["calibration_traces"]:
        max_turns = int(cfg["dataset"]["calibration_max_turns"])
        session = build_session(cfg, spec, args, trace_index=idx, max_turns=max_turns)
        opts = options_from_cfg(cfg, spec, "calibrate", max_turns)
        res = TurnRunner(model, opts).run_session(session, run_dir, None)
        for g, h in res["histograms"].items():
            histograms[g] = histograms.get(g, 0) + h
        sources.append({"trace_index": idx, "session_id": session.session_id, "turns": len(res["turn_rows"]),
                        "final_len": res["final_len"], "fingerprint": session_fingerprint(session)})
        write_csv(run_dir / f"calibration_turns_trace{idx}.csv", res["turn_rows"])
    tables = build_tables(histograms)
    save_tables(run_dir / "shared_tables.npz", tables, {"model": spec, "sources": sources,
                                                         "precision_bits": 15, "smoothing": "every symbol weight >= 1"})
    print(run_dir / "shared_tables.npz", table_set_bytes(tables), "bytes")


def cmd_run(cfg, spec, args, run_dir):
    device = torch.device(args.device)
    session = build_session(cfg, spec, args, max_turns=args.max_turns)
    opts = options_from_cfg(cfg, spec, "encode", args.max_turns)
    tables = {}
    if args.tables:
        tables = load_tables(Path(args.tables))
    elif opts.primary and opts.primary[0] == "ans_lane_shared":
        raise SystemExit("primary codec needs --tables (run calibrate first)")
    write_json(run_dir / "manifest.json", {
        "run_id": args.run_id, "model": spec, "source": args.source, "session_id": session.session_id,
        "source_kind": session.source_kind, "session_meta": session.meta, "turns_available": len(session.turns),
        "session_fingerprint": session_fingerprint(session), "options": vars(opts),
        "shared_tables": args.tables, "shared_table_set_bytes": table_set_bytes(tables) if tables else 0,
    })
    model = load_model(spec, device)
    started = time.monotonic()
    artifact_dir = ROOT / cfg["runtime"]["raw_artifact_root"] / args.run_id
    res = TurnRunner(model, opts, tables).run_session(session, run_dir, artifact_dir)
    turn_codec = agg_rows(res["agg"])
    summary = summarize(turn_codec)
    write_csv(run_dir / "turns.csv", res["turn_rows"])
    write_csv(run_dir / "turn_codec.csv", turn_codec)
    write_csv(run_dir / "summary.csv", summary)
    write_json(run_dir / "storage.json", storage_accounting(
        res["turn_rows"], summary, opts.primary, opts.cache_block_tokens, table_set_bytes(tables) if tables else 0))
    write_json(run_dir / "verification.json", {
        "bit_exact_units": res["bit_exact_units"],
        "all_units_bit_exact": True,  # encode_job raises on the first mismatch
        "rehydration": res.get("rehydration"),
        "truncated_reason": res["truncated_reason"],
        "turns_processed": len(res["turn_rows"]),
        "turns_available": len(session.turns),
        "wall_seconds": time.monotonic() - started,
    })
    print(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "calibrate", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "turns.yaml")
    parser.add_argument("--model", required=True, help="key under models: in the config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", choices=["agentx", "trajectory"], default="agentx")
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--trajectory-file")
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--tables", help="shared_tables.npz from a calibrate run of the same model")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--audit-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0", help="cpu only for smoke tests; results require CUDA")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    spec = dict(cfg["models"][args.model])
    spec["key"] = args.model
    spec["revision"] = resolve_revision(spec)
    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing to claim a real model forward")
    env_cfg = {"runtime": cfg["runtime"]}
    write_json(run_dir / "env_summary.json", {**environment_summary(env_cfg, sys.argv),
                                               "hf_home": os.environ.get("HF_HOME")})
    {"audit": cmd_audit, "calibrate": cmd_calibrate, "run": cmd_run}[args.command](cfg, spec, args, run_dir)


if __name__ == "__main__":
    main()
