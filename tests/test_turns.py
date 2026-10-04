"""Unit tests for the per-turn pipeline.

Random weights / random token IDs here are test fixtures only; they never
enter experiment results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_inputs import (  # noqa: E402
    Session,
    Turn,
    common_prefix_len,
    hash_id_block_keys,
    token_block_keys,
    trace_turn_stats,
)
from kv_codec import decode2, encode2, lane_histograms, order0_entropy_bytes, quantize_counts  # noqa: E402


# ---------------------------------------------------------------- codec v2


@pytest.mark.parametrize("codec", ["zstd", "ans_raw", "ans_lane", "ans_lane_shared"])
@pytest.mark.parametrize("element_size", [1, 2, 4])
@pytest.mark.parametrize("raw", [b"", b"\x00\x00\x00\x00", bytes(range(256)) * 4, b"\x00\x80\x7f\xff" * 257])
def test_codec2_roundtrip(codec, element_size, raw):
    if len(raw) % element_size:
        pytest.skip("length must be a multiple of the element size")
    lanes = 1 if codec in ("zstd", "ans_raw") else element_size
    # Shared table deliberately built from *different* data: unseen symbols must stay encodable.
    shared = np.stack([quantize_counts(h) for h in lane_histograms(b"\x01\x02" * 64, lanes)])
    enc = encode2(raw, codec=codec, element_size=element_size, shared_tables=shared)
    assert decode2(enc.container, element_size=element_size, shared_tables=shared) == raw
    assert enc.total_bytes == len(enc.container)
    assert enc.total_bytes == enc.payload_bytes + enc.table_bytes + enc.header_bytes


def test_codec2_truncation_rejected():
    enc = encode2(bytes(range(64)) * 8, codec="ans_lane", element_size=2)
    with pytest.raises(ValueError):
        decode2(enc.container[:-1], element_size=2)


def test_shared_table_has_no_inline_table():
    raw = bytes(range(256)) * 16
    shared = np.stack([quantize_counts(h) for h in lane_histograms(raw, 2)])
    inline = encode2(raw, codec="ans_lane", element_size=2)
    ext = encode2(raw, codec="ans_lane_shared", element_size=2, shared_tables=shared)
    assert inline.table_bytes == 2 * 256 * 2 and ext.table_bytes == 0


def test_quantize_counts_properties():
    w = quantize_counts(np.r_[np.zeros(255), 1e9])
    assert w.sum() == 1 << 15 and w.min() == 1


def test_entropy_matches_uniform():
    raw = bytes(range(256)) * 8
    assert order0_entropy_bytes(raw, 1) == pytest.approx(len(raw))


# ---------------------------------------------------------------- inputs


def test_chain_keys_are_prefix_sensitive():
    a = token_block_keys(np.arange(256, dtype=np.uint32), 64)
    b = np.arange(256, dtype=np.uint32)
    b[10] = 9999
    kb = token_block_keys(b, 64)
    assert len(a) == 4 and all(x != y for x, y in zip(a, kb))  # change in block 0 changes every key
    assert hash_id_block_keys([1, 2, 3])[:2] == hash_id_block_keys([1, 2, 9])[:2]


def test_common_prefix_len():
    assert common_prefix_len(np.array([1, 2, 3]), np.array([1, 2, 4, 5])) == 2
    assert common_prefix_len(np.array([], dtype=np.uint32), np.array([1])) == 0


def test_trace_turn_stats_dedup():
    row = {
        "id": "s",
        "block_size": 4,
        "hash_id_scope": "local",
        "requests": [
            {"type": "n", "t": 0.0, "in": 8, "out": 1, "hash_ids": [1, 2], "model": "m"},
            {"type": "subagent", "t": 1.0, "requests": [{"type": "n", "t": 1.0, "in": 4, "out": 1, "hash_ids": [7]}]},
            {"type": "s", "t": 2.0, "in": 14, "out": 1, "hash_ids": [1, 2, 3], "model": "m"},
            {"type": "n", "t": 3.0, "in": 8, "out": 1, "hash_ids": [1, 5], "model": "m"},
        ],
    }
    turns, summary = trace_turn_stats(row)
    assert [t["new_blocks"] for t in turns] == [2, 1, 1]
    assert [t["hit_blocks"] for t in turns] == [0, 2, 1]
    assert turns[1]["uncovered_tail_tokens"] == 2
    assert summary["snapshot_tokens"] == 30 and summary["unique_block_tokens"] == 16
    assert summary["subagent_requests"] == 1


# ---------------------------------------------------------------- tiny models (CPU)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from turn_runner import RunOptions, TurnRunner, agg_rows, build_tables  # noqa: E402


def _tiny_model(kind: str):
    torch.manual_seed(0)
    if kind == "hybrid":
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )
        model = transformers.Qwen3_5ForCausalLM(cfg)
    else:
        cfg = transformers.Qwen3Config(
            vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        )
        model = transformers.Qwen3ForCausalLM(cfg)
    return model.to(torch.bfloat16).eval()


def _session() -> Session:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 97, 40).astype(np.uint32)
    t0 = base[:20]  # 2 full blocks
    t1 = base[:33]  # extends t0
    t2 = np.r_[base[:16], rng.integers(0, 97, 16)].astype(np.uint32)  # diverges exactly at a snapshot (16)
    t3 = np.r_[t2, rng.integers(0, 97, 9)].astype(np.uint32)  # extends t2
    t4 = np.r_[base[:12], rng.integers(0, 97, 20)].astype(np.uint32)  # diverges inside block 1: no snapshot
    t5 = base[:40]  # back to the first branch: blocks 0-3 already stored, must not be re-stored
    turns = [Turn(i, f"main/{i}", t) for i, t in enumerate([t0, t1, t2, t3, t4, t5])]
    return Session("tiny", "test-fixture", turns)


@pytest.mark.parametrize("kind", ["hybrid", "standard"])
def test_turn_runner_end_to_end(kind, tmp_path):
    model = _tiny_model(kind)
    base = dict(cache_block_tokens=8, prefill_chunk_tokens=5, max_context_tokens=1024, max_snapshots=8,
                kv_formats=["bf16", "fp8_e4m3_sat"], cpu_workers=2)
    calib = TurnRunner(model, RunOptions(mode="calibrate", **base)).run_session(_session(), tmp_path / "c", None)
    tables = build_tables(calib["histograms"])
    opts = RunOptions(
        mode="encode",
        kv_codecs=[("zstd", 8), ("ans_raw", 8), ("ans_lane", 8), ("ans_lane_shared", 8), ("ans_lane", 16)],
        state_codecs=["zstd", "ans_lane", "ans_lane_shared"],
        primary=("ans_lane_shared", 8),
        **base,
    )
    res = TurnRunner(model, opts, tables).run_session(_session(), tmp_path / "r", tmp_path / "artifacts")
    rows = res["turn_rows"]
    assert [r["new_blocks"] for r in rows] == [2, 2, 2, 1, 3, 1]
    assert [r["hit_blocks"] for r in rows] == [0, 2, 2, 4, 1, 4]
    if kind == "hybrid":
        assert [r["restore"] for r in rows] == ["none", "none", "snapshot", "none", "reset", "reset"]
        assert rows[2]["rollback_to"] == 16 and rows[2]["fed_tokens"] == 16
        assert all(r["state_snapshot_raw_bytes"] > 0 for r in rows)
    else:
        assert [r["restore"] for r in rows] == ["none", "none", "crop", "none", "crop", "crop"]
        assert rows[4]["rollback_to"] == 8
        assert all(r["state_snapshot_raw_bytes"] == 0 for r in rows)
    reh = res["rehydration"]
    assert reh["status"] == "ok", reh
    assert reh["tokens"] == 40
    table = agg_rows(res["agg"])
    assert res["bit_exact_units"] == sum(r["units"] for r in table)
    kv_bf16 = [r for r in table if r["format"] == "bf16" and r["codec"] == "ans_lane" and r["block_tokens"] == 8]
    assert sum(r["raw_bytes"] for r in kv_bf16) == sum(r["new_kv_raw_bytes"] for r in rows)
    assert (tmp_path / "artifacts" / "units.csv.gz").exists()
