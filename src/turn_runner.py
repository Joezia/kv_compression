"""Per-turn persistent-state capture and lossless coding for agentic sessions.

Semantics (vLLM-style prefix caching, block = `cache_block_tokens` tokens):
- Each turn's input is processed with incremental prefill on top of the
  cached prefix. Only full cache blocks are cached; the partial tail of a turn
  is processed in a later turn (or never, for the last turn).
- A block is "new" if its prefix-chain key was never stored in this session.
  New full-attention K/V blocks are coded; previously stored blocks are not.
- Hybrid models: the linear-attention states (recurrent + conv) are snapshotted
  at the end of the last full block of every turn that produced new blocks.
  Those snapshots are stored and coded too (they cannot be deduplicated).
- When a turn diverges from the cached prefix, standard-attention caches are
  cropped; hybrid caches restore the latest compatible snapshot, otherwise
  they restart from position 0. Every rollback is logged.

Lossless is defined against the actual cache buffer bytes (BF16/FP32 as
produced by the model). The optional FP8 format is a labelled post-hoc
conversion of K/V (saturating cast to E4M3, scale 1.0, no feedback into the
model) and is lossless only relative to that FP8 buffer.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import multiprocessing
import struct
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agentic_inputs import Session, common_prefix_len, token_block_keys
from kv_codec import decode2, encode2, lane_histograms, order0_entropy_bytes, quantize_counts

FP8_MAX = 448.0
SUPPORTED_LAYER_TYPES = {"full_attention", "linear_attention"}


# --------------------------------------------------------------------------
# Model and cache helpers
# --------------------------------------------------------------------------


def load_model(spec: dict[str, Any], device: torch.device):
    import transformers

    loader = getattr(transformers, spec.get("loader", "AutoModelForCausalLM"))
    model = loader.from_pretrained(
        spec["id"],
        revision=spec.get("revision"),
        dtype=getattr(torch, spec.get("dtype", "bfloat16")),
        low_cpu_mem_usage=True,
        attn_implementation=spec.get("attn_implementation", "sdpa"),
    ).to(device)
    model.eval()
    return model


def decoder_of(model):
    return model.model if hasattr(model, "model") else model


def layer_types_of(model) -> list[str]:
    cfg = model.config.get_text_config()
    types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers)
    unknown = sorted(set(types) - SUPPORTED_LAYER_TYPES)
    if unknown:
        raise ValueError(f"unaudited layer types {unknown}; extend the runner before using this model")
    return types


def new_cache(model):
    from transformers import DynamicCache

    return DynamicCache(config=model.config)


def kv_tensors(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cache.layers[layer_idx]
    keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError(f"layer {layer_idx} has no K/V tensors")
    return keys, values


def linear_state_tensors(cache, layer_idx: int) -> list[tuple[str, int, torch.Tensor]]:
    layer = cache.layers[layer_idx]
    out = []
    for attr, kind in (("conv_states", "convolution"), ("recurrent_states", "recurrent")):
        mapping = getattr(layer, attr, None)
        if not isinstance(mapping, dict):
            raise TypeError(f"linear layer {layer_idx} lacks {attr}")
        for idx, tensor in sorted(mapping.items()):
            if tensor is not None:
                out.append((kind, int(idx), tensor))
    if not out:
        raise ValueError(f"linear layer {layer_idx} has no initialized state")
    return out


def cache_seq_len(cache, layer_types: list[str]) -> int:
    for idx, kind in enumerate(layer_types):
        if kind == "full_attention":
            layer = cache.layers[idx]
            keys = getattr(layer, "keys", None)
            if not isinstance(keys, torch.Tensor) or keys.numel() == 0:
                return 0
            return int(keys.shape[-2])
    raise ValueError("model has no full-attention layer")


def crop_kv(cache, layer_types: list[str], length: int) -> None:
    for idx, kind in enumerate(layer_types):
        if kind == "full_attention":
            layer = cache.layers[idx]
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]


def snapshot_linear_states(cache, layer_types: list[str]) -> dict[tuple[int, str, int], torch.Tensor]:
    snap = {}
    for idx, kind in enumerate(layer_types):
        if kind == "linear_attention":
            for state_kind, state_idx, tensor in linear_state_tensors(cache, idx):
                snap[(idx, state_kind, state_idx)] = tensor.detach().to("cpu", copy=True).contiguous()
    return snap


def restore_linear_states(cache, snap: dict[tuple[int, str, int], torch.Tensor]) -> None:
    with torch.inference_mode():  # cache tensors were created under inference mode
        for (idx, state_kind, state_idx), saved in snap.items():
            layer = cache.layers[idx]
            mapping = layer.conv_states if state_kind == "convolution" else layer.recurrent_states
            mapping[state_idx].copy_(saved.to(mapping[state_idx].device))


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


# --------------------------------------------------------------------------
# Coding work units (executed in worker processes)
# --------------------------------------------------------------------------

_TABLES: dict[str, np.ndarray] = {}


def _init_worker(tables: dict[str, np.ndarray]) -> None:
    global _TABLES
    _TABLES = tables


def table_group(fmt: str, state_kind: str, layer_idx: int) -> str:
    return f"{fmt}|{state_kind}|{layer_idx}"


@dataclass
class SpanJob:
    """A contiguous token span of one K/V tensor (or one whole state tensor).

    `raw` is laid out as [rows, n_tokens, row_bytes] (HTD for K/V: rows = KV
    heads, row_bytes = head_dim * element_size). Workers split it into codec
    units of `block_tokens` tokens; whole-state jobs use rows = n_tokens = 1.
    """

    turn_idx: int
    fmt: str  # bf16 | fp8_e4m3_sat | native
    state_kind: str
    layer_idx: int
    state_idx: int
    token_start: int | None
    rows: int
    n_tokens: int
    row_bytes: int
    element_size: int
    raw: bytes
    codecs_by_block: dict[int | None, list[str]]
    primary: tuple[str, int | None] | None = None  # (codec, block_tokens) that goes to the archive
    primary_keys: list[tuple] | None = None  # one archive key per primary unit


def _job_units(job: SpanJob):
    if job.token_start is None:  # whole-state tensor
        yield None, None, None, job.raw, 0
        return
    arr = np.frombuffer(job.raw, dtype=np.uint8).reshape(job.rows, job.n_tokens, job.row_bytes)
    for block_tokens in job.codecs_by_block:
        for unit_idx, a in enumerate(range(0, job.n_tokens, block_tokens)):
            b = min(a + block_tokens, job.n_tokens)
            yield block_tokens, job.token_start + a, job.token_start + b, arr[:, a:b, :].tobytes(), unit_idx


def encode_job(job: SpanJob, zstd_level: int) -> list[dict]:
    """Encode every unit of a job with every configured codec; verify bit-exact."""
    rows = []
    group = table_group(job.fmt, job.state_kind, job.layer_idx)
    for block_tokens, t0, t1, raw, unit_idx in _job_units(job):
        entropy_lane = order0_entropy_bytes(raw, job.element_size)
        entropy_raw = order0_entropy_bytes(raw, 1) if job.element_size > 1 else entropy_lane
        digest = hashlib.sha256(raw).hexdigest()
        for codec in job.codecs_by_block[block_tokens]:
            if codec == "ans_raw" and job.element_size == 1:
                continue  # identical to ans_lane for 1-byte elements
            tables = None
            if codec == "ans_lane_shared":
                tables = _TABLES.get(group)
                if tables is None:
                    continue
            enc = encode2(raw, codec=codec, element_size=job.element_size, shared_tables=tables, zstd_level=zstd_level)
            if decode2(enc.container, element_size=job.element_size, shared_tables=tables) != raw:
                raise AssertionError(f"non-bit-exact unit {group} turn {job.turn_idx} {codec} {block_tokens}")
            is_primary = job.primary is not None and job.primary == (codec, block_tokens)
            rows.append(
                {
                    "turn_idx": job.turn_idx,
                    "format": job.fmt,
                    "state_kind": job.state_kind,
                    "layer_idx": job.layer_idx,
                    "state_idx": job.state_idx,
                    "token_start": t0,
                    "token_end": t1,
                    "block_tokens": block_tokens,
                    "element_size": job.element_size,
                    "codec": codec,
                    "raw_bytes": len(raw),
                    "payload_bytes": enc.payload_bytes,
                    "table_bytes": enc.table_bytes,
                    "header_bytes": enc.header_bytes,
                    "compressed_total_bytes": enc.total_bytes,
                    "entropy_lane_bytes": entropy_lane,
                    "entropy_raw_bytes": entropy_raw,
                    "raw_sha256": digest,
                    "bit_exact": True,
                    "_container": enc.container if is_primary else None,
                    "_primary_key": job.primary_keys[unit_idx] if is_primary else None,
                }
            )
    return rows


# --------------------------------------------------------------------------
# Session runner
# --------------------------------------------------------------------------


@dataclass
class RunOptions:
    cache_block_tokens: int = 64
    prefill_chunk_tokens: int = 8192
    max_context_tokens: int = 131072
    max_turns: int | None = None
    max_snapshots: int = 8
    kv_formats: list[str] = field(default_factory=lambda: ["bf16"])
    kv_codecs: list[tuple[str, int]] = field(default_factory=list)  # (codec, block_tokens)
    state_codecs: list[str] = field(default_factory=list)
    primary: tuple[str, int] | None = None
    zstd_level: int = 3
    cpu_workers: int = 8
    max_inflight_bytes: int = 4 << 30
    max_gpu_memory_bytes: int | None = None
    mode: str = "encode"  # encode | calibrate


class TurnRunner:
    def __init__(self, model, opts: RunOptions, tables: dict[str, np.ndarray] | None = None):
        self.model = model
        self.decoder = decoder_of(model)
        self.opts = opts
        self.layer_types = layer_types_of(model)
        self.tables = tables or {}
        self.device = next(model.parameters()).device
        self.is_hybrid = "linear_attention" in self.layer_types
        if opts.mode == "encode" and opts.primary is not None:
            codec, block = opts.primary
            if block != opts.cache_block_tokens:
                raise ValueError("primary codec unit must equal the cache block size")
            if (codec, block) not in [tuple(c) for c in opts.kv_codecs]:
                raise ValueError("primary config must be listed in kv_codecs")
            if codec == "ans_lane_shared" and not self.tables:
                raise ValueError("primary uses shared tables but none were loaded")

    # -- forward -------------------------------------------------------------
    def _feed(self, cache, token_ids: np.ndarray) -> float:
        started = time.monotonic()
        chunk = self.opts.prefill_chunk_tokens
        with torch.inference_mode():
            for s in range(0, len(token_ids), chunk):
                ids = torch.from_numpy(token_ids[s : s + chunk].astype(np.int64)).unsqueeze(0).to(self.device)
                self.decoder(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.monotonic() - started

    # -- job construction -----------------------------------------------------
    def _kv_jobs(self, cache, turn_idx: int, start: int, end: int, block_keys: list[bytes], stats: dict):
        """Yield one SpanJob per (layer, K/V, format) for the new token span [start, end)."""
        B = self.opts.cache_block_tokens
        codecs_by_block: dict[int | None, list[str]] = defaultdict(list)
        for codec, block_tokens in self.opts.kv_codecs:
            codecs_by_block[int(block_tokens)].append(codec)
        if self.opts.mode == "calibrate":
            codecs_by_block = {B: []}
        n = end - start
        for layer_idx, kind in enumerate(self.layer_types):
            if kind != "full_attention":
                continue
            keys, values = kv_tensors(cache, layer_idx)
            for state_kind, tensor in (("key", keys), ("value", values)):
                piece = tensor[0, :, start:end, :]
                rows, _, head_dim = piece.shape
                for fmt in self.opts.kv_formats:
                    if fmt == "bf16":
                        if piece.dtype != torch.bfloat16:
                            raise ValueError(f"expected bf16 K/V, got {piece.dtype}")
                        raw = tensor_bytes(piece)
                        element_size = 2
                        stats["kv_raw_bytes_bf16"] += len(raw)
                    elif fmt == "fp8_e4m3_sat":
                        f = piece.float()
                        stats["fp8_clamped_elements"] += int((f.abs() > FP8_MAX).sum())
                        raw = tensor_bytes(f.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn))
                        element_size = 1
                    else:
                        raise ValueError(f"unknown kv format {fmt}")
                    primary = primary_keys = None
                    if fmt == "bf16" and self.opts.primary is not None and self.opts.mode == "encode":
                        primary = (self.opts.primary[0], int(self.opts.primary[1]))
                        primary_keys = [("kv", block_keys[(start + a) // B], layer_idx, state_kind) for a in range(0, n, B)]
                    yield SpanJob(turn_idx, fmt, state_kind, layer_idx, 0, start, rows, n, head_dim * element_size,
                                  element_size, raw, dict(codecs_by_block), primary, primary_keys)

    def _state_jobs(self, snap: dict, turn_idx: int, state_key: bytes):
        codecs = {None: list(self.opts.state_codecs) if self.opts.mode == "encode" else []}
        primary = None
        if self.opts.primary is not None and self.opts.mode == "encode":
            primary = (self.opts.primary[0], None)
            if primary[0] not in self.opts.state_codecs:
                raise ValueError("primary codec must also be listed in state_codecs for hybrid models")
        for (layer_idx, state_kind, state_idx), tensor in sorted(snap.items()):
            raw = tensor_bytes(tensor)
            yield SpanJob(turn_idx, "native", state_kind, layer_idx, state_idx, None, 1, 1, len(raw),
                          tensor.element_size(), raw, codecs, primary,
                          [("state", state_key, layer_idx, state_kind, state_idx)])

    # -- main loop ------------------------------------------------------------
    def run_session(self, session: Session, out_dir: Path, artifact_dir: Path | None) -> dict[str, Any]:
        opts = self.opts
        B = opts.cache_block_tokens
        cache = new_cache(self.model)
        cur_tokens = np.zeros(0, dtype=np.uint32)
        cur_len = 0
        # (position, prefix-chain key at that position) -> linear-state snapshot
        snapshots: OrderedDict[tuple[int, bytes | None], dict] = OrderedDict()
        seen: set[bytes] = set()
        turn_rows: list[dict] = []
        histograms: dict[str, np.ndarray] = {}
        agg: dict[tuple, list] = defaultdict(lambda: [0] * 8)
        bit_exact_units = 0
        primary_index: dict[tuple, tuple[int, int]] = {}
        archive = None
        units_file = None
        units_writer = None
        if opts.mode == "encode":
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=False)
                if opts.primary is not None:
                    archive = (artifact_dir / f"primary_{opts.primary[0]}_{opts.primary[1]}.akv2").open("xb")
                units_file = gzip.open(artifact_dir / "units.csv.gz", "wt", newline="", encoding="utf-8")
        # spawn, not fork: the parent holds a CUDA context; workers only run CPU codecs.
        pool = ProcessPoolExecutor(
            max_workers=opts.cpu_workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(self.tables,),
        ) if opts.mode == "encode" else None
        pending: list[tuple[Future, int]] = []
        inflight = 0
        truncated_reason = None

        def collect(fut: Future) -> None:
            nonlocal bit_exact_units, units_writer
            for row in fut.result():
                container = row.pop("_container")
                pkey = row.pop("_primary_key")
                if container is not None and archive is not None:
                    offset = archive.tell()
                    archive.write(struct.pack("<Q", len(container)))
                    archive.write(container)
                    primary_index[pkey] = (offset, len(container))
                bit_exact_units += 1
                key = (row["turn_idx"], row["format"], row["state_kind"], row["codec"], row["block_tokens"])
                a = agg[key]
                a[0] += 1
                a[1] += row["raw_bytes"]
                a[2] += row["payload_bytes"]
                a[3] += row["table_bytes"]
                a[4] += row["header_bytes"]
                a[5] += row["compressed_total_bytes"]
                a[6] += row["entropy_lane_bytes"]
                a[7] += row["entropy_raw_bytes"]
                if units_file is not None:
                    if units_writer is None:
                        units_writer = csv.DictWriter(units_file, fieldnames=list(row))
                        units_writer.writeheader()
                    units_writer.writerow(row)

        def submit(jobs) -> int:
            nonlocal inflight
            total = 0
            for job in jobs:
                total += len(job.raw)
                if opts.mode == "calibrate":
                    group = table_group(job.fmt, job.state_kind, job.layer_idx)
                    histograms[group] = histograms.get(group, 0) + lane_histograms(job.raw, job.element_size)
                    continue
                pending.append((pool.submit(encode_job, job, opts.zstd_level), len(job.raw)))
                inflight += len(job.raw)
                while inflight > opts.max_inflight_bytes and pending:
                    fut, sz = pending.pop(0)
                    collect(fut)
                    inflight -= sz
            return total

        try:
            for turn in session.turns:
                if opts.max_turns is not None and turn.turn_idx >= opts.max_turns:
                    truncated_reason = f"max_turns={opts.max_turns}"
                    break
                x = np.asarray(turn.token_ids, dtype=np.uint32)
                nfull = len(x) // B
                end = nfull * B
                if end > opts.max_context_tokens:
                    truncated_reason = f"turn {turn.turn_idx} needs {end} tokens > max_context_tokens={opts.max_context_tokens}"
                    break
                keys = token_block_keys(x, B)
                lcp = common_prefix_len(cur_tokens, x[:end])
                target = (lcp // B) * B
                restore = "none"
                rollback_from = cur_len
                if target < cur_len:
                    if not self.is_hybrid:
                        crop_kv(cache, self.layer_types, target)
                        cur_len = target
                        restore = "crop"
                    else:
                        candidates = [(p, k) for p, k in snapshots if 0 < p <= target and k == keys[p // B - 1]]
                        if candidates:
                            best = max(candidates, key=lambda pk: pk[0])
                            pos = best[0]
                            crop_kv(cache, self.layer_types, pos)
                            restore_linear_states(cache, snapshots[best])
                            snapshots.move_to_end(best)
                            cur_len = pos
                            restore = "snapshot"
                        else:
                            cache = new_cache(self.model)
                            cur_len = 0
                            restore = "reset"
                    cur_tokens = cur_tokens[:cur_len]
                fed = max(0, end - cur_len)
                forward_s = self._feed(cache, x[cur_len:end]) if fed else 0.0
                cur_tokens = x[:end].copy()
                cur_len = end
                if cur_len and cache_seq_len(cache, self.layer_types) != cur_len:
                    raise AssertionError("cache length does not match the tracked prefix")
                peak = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0
                if opts.max_gpu_memory_bytes and peak > opts.max_gpu_memory_bytes:
                    truncated_reason = f"GPU reserved {peak} > limit at turn {turn.turn_idx}"
                    break
                first_new = nfull
                for i, key in enumerate(keys):
                    if key not in seen:
                        first_new = i
                        break
                seen.update(keys)
                new_start = first_new * B
                snap_bytes = 0
                kv_stats = {"fp8_clamped_elements": 0, "kv_raw_bytes_bf16": 0}
                if self.is_hybrid and end > 0:
                    snap_id = (end, keys[nfull - 1] if nfull else None)
                    snapshots[snap_id] = snapshot_linear_states(cache, self.layer_types)
                    snapshots.move_to_end(snap_id)
                    while len(snapshots) > opts.max_snapshots:
                        snapshots.popitem(last=False)
                if end > new_start:
                    submit(self._kv_jobs(cache, turn.turn_idx, new_start, end, keys, kv_stats))
                    if self.is_hybrid:
                        snap_bytes = submit(self._state_jobs(snapshots[snap_id], turn.turn_idx, keys[nfull - 1]))
                turn_rows.append(
                    {
                        "session_id": session.session_id,
                        "turn_idx": turn.turn_idx,
                        "request_path": turn.request_path,
                        "t": turn.meta.get("t"),
                        "think_time": turn.meta.get("think_time"),
                        "input_tokens": len(x),
                        "full_blocks": nfull,
                        "model_lcp_tokens": lcp,
                        "restore": restore,
                        "rollback_from": rollback_from if restore != "none" else None,
                        "rollback_to": cur_len - fed if restore != "none" else None,
                        "fed_tokens": fed,
                        "forward_seconds": forward_s,
                        "hit_blocks": first_new,
                        "new_blocks": nfull - first_new,
                        "new_kv_raw_bytes": kv_stats["kv_raw_bytes_bf16"],
                        "state_snapshot_raw_bytes": snap_bytes,
                        "fp8_clamped_elements": kv_stats["fp8_clamped_elements"],
                        "uncovered_tail_tokens": len(x) - end,
                        "gpu_peak_reserved_bytes": peak,
                    }
                )
            for fut, _ in pending:
                collect(fut)
            pending.clear()
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            if archive is not None:
                archive.close()
            if units_file is not None:
                units_file.close()

        result = {
            "turn_rows": turn_rows,
            "truncated_reason": truncated_reason,
            "bit_exact_units": bit_exact_units,
            "agg": agg,
            "histograms": histograms,
            "final_len": cur_len,
        }
        final_snapshot = None
        if self.is_hybrid and cur_len > 0:
            fid = (cur_len, keys_for(cur_tokens, B)[-1])
            final_snapshot = (fid[1], snapshots[fid]) if fid in snapshots else None
        if opts.mode == "encode" and archive is not None and cur_len > 0:
            result["rehydration"] = self._rehydration_check(
                cache, cur_tokens, keys_for(cur_tokens, B), primary_index, artifact_dir, opts.primary,
                final_snapshot,
            )
        return result

    # -- end-to-end check -----------------------------------------------------
    def _rehydration_check(self, *args) -> dict:
        """Rebuild the final prefix from the primary archive and compare with the live cache."""
        with torch.inference_mode():
            return self._rehydration_check_impl(*args)

    def _rehydration_check_impl(self, live_cache, tokens, block_keys, index, artifact_dir, primary, final_snapshot) -> dict:
        codec, B = primary
        path = artifact_dir / f"primary_{codec}_{B}.akv2"
        data = path.read_bytes()

        def fetch(key, element_size, group):
            offset, length = index[key]
            (stored_len,) = struct.unpack_from("<Q", data, offset)
            if stored_len != length:
                raise ValueError("archive index mismatch")
            container = data[offset + 8 : offset + 8 + length]
            return decode2(container, element_size=element_size, shared_tables=self.tables.get(group))

        rebuilt = new_cache(self.model)
        mismatched_blocks = 0
        missing = 0
        missing_state = 0
        for layer_idx, kind in enumerate(self.layer_types):
            if kind == "full_attention":
                live_k, live_v = kv_tensors(live_cache, layer_idx)
                parts = {}
                for state_kind, live in (("key", live_k), ("value", live_v)):
                    pieces = []
                    for i, bkey in enumerate(block_keys):
                        key = ("kv", bkey, layer_idx, state_kind)
                        if key not in index:
                            missing += 1
                            continue
                        raw = fetch(key, live.element_size(), table_group("bf16", state_kind, layer_idx))
                        if raw != tensor_bytes(live[0, :, i * B : (i + 1) * B, :]):
                            mismatched_blocks += 1
                        h, d = live.shape[1], live.shape[3]
                        pieces.append(torch.frombuffer(bytearray(raw), dtype=live.dtype).reshape(1, h, B, d))
                    parts[state_kind] = torch.cat(pieces, dim=2).to(live.device) if pieces else None
                if parts["key"] is not None:
                    rebuilt.update(parts["key"], parts["value"], layer_idx)
        state_mismatch = 0
        if self.is_hybrid:
            if final_snapshot is None:
                return {"status": "skipped", "reason": "final linear-state snapshot evicted"}
            state_key, snap = final_snapshot
            for (layer_idx, state_kind, state_idx), saved in sorted(snap.items()):
                key = ("state", state_key, layer_idx, state_kind, state_idx)
                if key not in index:
                    missing_state += 1
                    continue
                raw = fetch(key, saved.element_size(), table_group("native", state_kind, layer_idx))
                if raw != tensor_bytes(saved):
                    state_mismatch += 1
                tensor = torch.frombuffer(bytearray(raw), dtype=saved.dtype).reshape(saved.shape).to(self.device)
                layer = live_cache.layers[layer_idx]
                if state_kind == "convolution":
                    rebuilt.update_conv_state(tensor, layer_idx, state_idx=state_idx,
                                              conv_kernel_size=int(layer.conv_kernel_size[state_idx]))
                else:
                    rebuilt.update_recurrent_state(tensor, layer_idx, state_idx=state_idx)
        if missing:
            return {"status": "failed", "missing_kv_units": missing}
        if missing_state:
            return {"status": "skipped", "reason": "final linear-state snapshot was not stored (no new blocks)"}
        nxt = torch.tensor([[int(tokens[-1])]], device=self.device)
        with torch.inference_mode():
            a = self.decoder(input_ids=nxt, past_key_values=live_cache, use_cache=True).last_hidden_state.float().cpu()
            b = self.decoder(input_ids=nxt, past_key_values=rebuilt, use_cache=True).last_hidden_state.float().cpu()
        return {
            "status": "ok",
            "tokens": len(tokens),
            "kv_blocks_differing_from_live_cache": mismatched_blocks,
            "state_tensors_differing_from_snapshot": state_mismatch,
            "continuation_hidden_exact": bool(torch.equal(a, b)),
            "continuation_hidden_max_abs_diff": float((a - b).abs().max()),
            "note": "blocks recomputed after a rollback may legitimately differ in bits from the stored original",
        }


def keys_for(tokens: np.ndarray, block_tokens: int) -> list[bytes]:
    return token_block_keys(tokens, block_tokens)


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------

AGG_FIELDS = ["units", "raw_bytes", "payload_bytes", "table_bytes", "header_bytes",
              "compressed_total_bytes", "entropy_lane_bytes", "entropy_raw_bytes"]


def agg_rows(agg: dict[tuple, list]) -> list[dict]:
    rows = []
    for (turn_idx, fmt, state_kind, codec, block_tokens), values in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {"turn_idx": turn_idx, "format": fmt, "state_kind": state_kind, "codec": codec, "block_tokens": block_tokens}
        row.update(dict(zip(AGG_FIELDS, values)))
        rows.append(row)
    return rows


def build_tables(histograms: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {group: np.stack([quantize_counts(h) for h in hist]) for group, hist in histograms.items()}


def table_set_bytes(tables: dict[str, np.ndarray]) -> int:
    return int(sum(t.size * 2 for t in tables.values()))


def save_tables(path: Path, tables: dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k.replace("|", "__"): v for k, v in tables.items()})
    path.with_suffix(".json").write_text(json.dumps({**meta, "table_set_bytes": table_set_bytes(tables),
                                                     "groups": sorted(tables)}, indent=2) + "\n")


def load_tables(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k.replace("__", "|"): data[k].astype(np.uint16) for k in data.files}
