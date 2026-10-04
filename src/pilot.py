from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import DynamicCache, Qwen3_5ForCausalLM

from kv_codec import encode_block, sha256_bytes


@dataclass
class PersistentState:
    state_id: str
    layer_idx: int
    layer_type: str
    state_kind: str
    state_idx: int
    dtype: str
    shape: list[int]
    original_stride: list[int]
    element_size: int
    raw: bytes
    state_scope: str
    token_span_start: int | None
    token_span_end: int | None
    conv_kernel_size: int | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "layer_idx": self.layer_idx,
            "layer_type": self.layer_type,
            "state_kind": self.state_kind,
            "state_idx": self.state_idx,
            "dtype": self.dtype,
            "shape": self.shape,
            "original_stride": self.original_stride,
            "encoded_layout": "contiguous_c_order",
            "endianness": sys.byteorder,
            "element_size": self.element_size,
            "raw_bytes": len(self.raw),
            "sha256": sha256_bytes(self.raw),
            "state_scope": self.state_scope,
            "token_span_start": self.token_span_start,
            "token_span_end": self.token_span_end,
            "conv_kernel_size": self.conv_kernel_size,
        }


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _drop_arrow_union_nulls(value: Any) -> Any:
    """Remove only Arrow-added null fields before strict Weka validation.

    Hugging Face's union struct materialization pads every request variant with
    the other variants' fields set to null. AIPerf's Weka Pydantic models use
    extra="forbid". Removing nulls is equivalent to the source JSON's absent
    optional/foreign fields; non-null unknown fields remain and still fail.
    """
    if isinstance(value, dict):
        return {
            key: _drop_arrow_union_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_arrow_union_nulls(item) for item in value]
    return value


def environment_summary(cfg: dict[str, Any], command: list[str]) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        gpu = {
            "logical_index": 0,
            "physical_index": cfg["runtime"]["physical_gpu"],
            "name": props.name,
            "total_bytes": total,
            "free_bytes_at_start": free,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "command": command,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "gpu": gpu,
        "packages": {
            name: _package_version(name)
            for name in [
                "torch",
                "transformers",
                "tokenizers",
                "datasets",
                "aiperf",
                "constriction",
                "zstandard",
                "numpy",
                "PyYAML",
            ]
        },
    }


def prepare_agentx_samples(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    from aiperf.common import random_generator as aiperf_rng
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.dataset.content import PromptConfig
    from aiperf.dataset.generator.corpus import resolve_prompt_generator
    from aiperf.dataset.loader.weka_trace_models import (
        WekaNormalRequest,
        WekaStreamingRequest,
        WekaTrace,
    )

    aiperf_rng.init(int(cfg["seed"]))
    data_cfg = cfg["dataset"]
    cache_dir = root / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    specs = list(data_cfg["samples"])
    trace_indices = sorted({int(spec["trace_index"]) for spec in specs})
    missing_indices = [
        index
        for index in trace_indices
        if not (cache_dir / f"agentx_trace_{index}.json").exists()
    ]
    if missing_indices:
        dataset = load_dataset(
            data_cfg["id"],
            split=data_cfg["split"],
            revision=data_cfg["revision"],
            streaming=True,
        )
        for index, row in enumerate(dataset):
            if index in missing_indices:
                (cache_dir / f"agentx_trace_{index}.json").write_text(
                    json.dumps(row, sort_keys=True), encoding="utf-8"
                )
            if index >= max(missing_indices):
                break

    tokenizer = Tokenizer.from_pretrained(
        cfg["model"]["id"],
        revision=cfg["model"]["revision"],
        resolve_alias=False,
    )
    prompt_generator = resolve_prompt_generator(
        corpus=data_cfg["corpus"],
        default_corpus=data_cfg["corpus"],
        tokenizer=tokenizer,
        prompts=PromptConfig(),
    )
    outputs: list[dict[str, Any]] = []
    for trace_index in trace_indices:
        row = json.loads(
            (cache_dir / f"agentx_trace_{trace_index}.json").read_text(encoding="utf-8")
        )
        trace = WekaTrace.model_validate(_drop_arrow_union_nulls(row))
        prompt_generator._cache.clear()
        prompt_generator._hash_id_corpus_rng.set_trace_id(trace.id)
        trace_specs = sorted(
            (spec for spec in specs if int(spec["trace_index"]) == trace_index),
            key=lambda spec: int(spec["outer_request_index"]),
        )
        for spec in trace_specs:
            stage = str(spec["stage"])
            outer_idx = int(spec["outer_request_index"])
            used_tokens = int(spec["used_tokens"])
            req = trace.requests[outer_idx]
            if not isinstance(req, WekaNormalRequest | WekaStreamingRequest):
                raise TypeError(f"outer request {outer_idx} is not a normal/streaming request")
            full_ids = prompt_generator._build_token_sequence(
                req.input_length, req.hash_ids, trace.block_size
            )
            if len(full_ids) != req.input_length:
                raise AssertionError("AIPerf synthesis did not produce the recorded input length")
            if used_tokens > len(full_ids):
                raise ValueError(f"requested {used_tokens} tokens from a {len(full_ids)} token request")
            ids_array = np.asarray(full_ids[:used_tokens], dtype="<u4")
            ids_path = cache_dir / f"{stage}_{trace.id}_outer{outer_idx}_{used_tokens}.u32"
            ids_path.write_bytes(ids_array.tobytes())
            record = {
                "stage": stage,
                "dataset_id": data_cfg["id"],
                "dataset_revision": data_cfg["revision"],
                "trace_index": trace_index,
                "trace_id": trace.id,
                "request_path": f"main/{outer_idx}",
                "outer_request_index": outer_idx,
                "request_type": req.type,
                "trace_model": req.model,
                "hash_id_scope": trace.hash_id_scope,
                "block_size_tokens": trace.block_size,
                "hash_ids": req.hash_ids,
                "hash_ids_sha256": hashlib.sha256(
                    np.asarray(req.hash_ids, dtype="<i8").tobytes()
                ).hexdigest(),
                "recorded_input_tokens": req.input_length,
                "used_input_tokens": used_tokens,
                "sampling_ratio": used_tokens / req.input_length,
                "source_kind": "AgentX-prefix-sampled",
                "input_mapping": "AIPerf-0.13.0 coding corpus hash-id token synthesis; direct token IDs; no chat-template round trip",
                "schema_adapter": "recursively drop only Arrow union padding fields whose value is null before strict WekaTrace validation",
                "input_ids_path": str(ids_path.relative_to(root)),
                "input_ids_sha256": hashlib.sha256(ids_array.tobytes()).hexdigest(),
                "tokenizer_id": cfg["model"]["id"],
                "tokenizer_revision": cfg["model"]["revision"],
                "tool_tokens": trace.tool_tokens,
                "system_tokens": trace.system_tokens,
            }
            _write_json(cache_dir / f"{stage}_sample.json", record)
            outputs.append(record)
    return outputs


def load_prepared_sample(cfg: dict[str, Any], root: Path, stage: str) -> tuple[dict[str, Any], torch.Tensor]:
    sample_path = root / "data" / "raw" / f"{stage}_sample.json"
    if not sample_path.exists():
        prepare_agentx_samples(cfg, root)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    raw = (root / sample["input_ids_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sample["input_ids_sha256"]:
        raise ValueError("prepared input_ids checksum mismatch")
    ids = np.frombuffer(raw, dtype="<u4").astype(np.int64)
    if len(ids) != sample["used_input_tokens"]:
        raise ValueError("prepared input_ids length mismatch")
    return sample, torch.from_numpy(ids.copy()).unsqueeze(0)


def _tensor_to_state(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    layer_type: str,
    state_kind: str,
    state_idx: int,
    token_count: int,
    conv_kernel_size: int | None = None,
) -> PersistentState:
    original_stride = list(tensor.stride())
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    is_kv = state_kind in {"key", "value"}
    return PersistentState(
        state_id=f"layer{layer_idx:02d}.{state_kind}.{state_idx}",
        layer_idx=layer_idx,
        layer_type=layer_type,
        state_kind=state_kind,
        state_idx=state_idx,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=list(tensor.shape),
        original_stride=original_stride,
        element_size=tensor.element_size(),
        raw=raw,
        state_scope="token_span" if is_kv else "terminal_after_prefill",
        token_span_start=0 if is_kv else None,
        token_span_end=token_count if is_kv else None,
        conv_kernel_size=conv_kernel_size,
    )


def extract_persistent_states(
    cache: DynamicCache, layer_types: list[str], token_count: int
) -> list[PersistentState]:
    if len(cache.layers) != len(layer_types):
        raise ValueError(f"cache/config layer mismatch: {len(cache.layers)} != {len(layer_types)}")
    states: list[PersistentState] = []
    for layer_idx, (layer, layer_type) in enumerate(zip(cache.layers, layer_types, strict=True)):
        if layer_type == "full_attention":
            for attr, kind in [("keys", "key"), ("values", "value")]:
                tensor = getattr(layer, attr, None)
                if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                    raise ValueError(f"full-attention layer {layer_idx} has no initialized {attr}")
                states.append(
                    _tensor_to_state(
                        tensor,
                        layer_idx=layer_idx,
                        layer_type=layer_type,
                        state_kind=kind,
                        state_idx=0,
                        token_count=token_count,
                    )
                )
        elif layer_type == "linear_attention":
            known_tensor_ids: set[int] = set()
            for attr, kind in [("conv_states", "convolution"), ("recurrent_states", "recurrent")]:
                mapping = getattr(layer, attr, None)
                if not isinstance(mapping, dict):
                    raise TypeError(f"linear-attention layer {layer_idx} has invalid {attr}")
                initialized = 0
                for state_idx, tensor in sorted(mapping.items()):
                    if tensor is None:
                        continue
                    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                        raise TypeError(f"invalid tensor in layer {layer_idx} {attr}[{state_idx}]")
                    initialized += 1
                    known_tensor_ids.add(id(tensor))
                    kernel = None
                    if kind == "convolution":
                        kernel = int(layer.conv_kernel_size[state_idx])
                    states.append(
                        _tensor_to_state(
                            tensor,
                            layer_idx=layer_idx,
                            layer_type=layer_type,
                            state_kind=kind,
                            state_idx=int(state_idx),
                            token_count=token_count,
                            conv_kernel_size=kernel,
                        )
                    )
                if initialized == 0:
                    raise ValueError(f"linear-attention layer {layer_idx} has no initialized {attr}")
            for value in vars(layer).values():
                if isinstance(value, torch.Tensor) and value.numel() and id(value) not in known_tensor_ids:
                    raise ValueError(f"unknown persistent tensor on linear-attention layer {layer_idx}")
        else:
            raise ValueError(f"unsupported non-audited layer type {layer_type!r} at layer {layer_idx}")
    return states


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
        "uint8": torch.uint8,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported reconstructed dtype: {name}") from error


def _bytes_to_tensor(state: PersistentState, raw: bytes, device: torch.device) -> torch.Tensor:
    tensor = torch.frombuffer(bytearray(raw), dtype=_torch_dtype(state.dtype)).clone()
    return tensor.reshape(state.shape).to(device)


def rebuild_cache(
    model_config: Any,
    states: list[PersistentState],
    decoded: dict[str, bytes],
    device: torch.device,
) -> DynamicCache:
    cache = DynamicCache(config=model_config)
    by_layer: dict[int, dict[str, PersistentState]] = defaultdict(dict)
    for state in states:
        by_layer[state.layer_idx][state.state_kind] = state
    for layer_idx, grouped in sorted(by_layer.items()):
        if "key" in grouped or "value" in grouped:
            key = grouped["key"]
            value = grouped["value"]
            cache.update(
                _bytes_to_tensor(key, decoded[key.state_id], device),
                _bytes_to_tensor(value, decoded[value.state_id], device),
                layer_idx,
            )
        else:
            conv = grouped["convolution"]
            recurrent = grouped["recurrent"]
            cache.update_conv_state(
                _bytes_to_tensor(conv, decoded[conv.state_id], device),
                layer_idx,
                state_idx=conv.state_idx,
                conv_kernel_size=conv.conv_kernel_size,
            )
            cache.update_recurrent_state(
                _bytes_to_tensor(recurrent, decoded[recurrent.state_id], device),
                layer_idx,
                state_idx=recurrent.state_idx,
            )
    return cache


def _write_blocks_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_blocks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scopes = ["complete_persistent_state", row["state_kind"]]
        if row["state_kind"] in {"key", "value"}:
            scopes.append("kv_only")
        for scope in scopes:
            groups[(row["codec"], row["block_size"], scope)].append(row)
    summaries: list[dict[str, Any]] = []
    for (codec, block_size, scope), items in sorted(groups.items()):
        raw_bytes = sum(int(x["raw_bytes"]) for x in items)
        compressed = sum(int(x["compressed_total_bytes"]) for x in items)
        summaries.append(
            {
                "codec": codec,
                "block_size": block_size,
                "scope": scope,
                "blocks": len(items),
                "raw_bytes": raw_bytes,
                "compressed_total_bytes": compressed,
                "compression_ratio": raw_bytes / compressed,
                "space_saving": 1.0 - compressed / raw_bytes,
                "bit_exact_blocks": sum(x["bit_exact"] == "true" for x in items),
            }
        )
    return summaries


def encode_states(
    *,
    states: list[PersistentState],
    codecs: list[str],
    block_sizes: list[int],
    zstd_level: int,
    artifact_dir: Path,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes], tuple[str, int]]:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    verification_choice = ("ans_raw", max(block_sizes))
    decoded_for_rebuild: dict[str, bytes] = {}
    for codec in codecs:
        for block_size in block_sizes:
            archive_path = artifact_dir / f"{codec}_{block_size}.akvpack"
            with archive_path.open("xb") as archive:
                decoded_by_state: dict[str, bytearray] = {
                    state.state_id: bytearray() for state in states
                }
                for state in states:
                    for block_idx, offset in enumerate(range(0, len(state.raw), block_size)):
                        raw = state.raw[offset : offset + block_size]
                        metadata = {
                            "run_id": run_id,
                            "state_id": state.state_id,
                            "layer_idx": state.layer_idx,
                            "layer_type": state.layer_type,
                            "state_kind": state.state_kind,
                            "state_idx": state.state_idx,
                            "dtype": state.dtype,
                            "shape": state.shape,
                            "tensor_byte_offset": offset,
                            "block_index": block_idx,
                            "block_size": block_size,
                        }
                        encoded = encode_block(
                            raw,
                            codec=codec,
                            element_size=state.element_size,
                            metadata=metadata,
                            zstd_level=zstd_level,
                        )
                        bit_exact = encoded.decoded == raw
                        if not bit_exact:
                            raise AssertionError(f"non-bit-exact block: {metadata}")
                        decoded_by_state[state.state_id].extend(encoded.decoded)
                        frame_offset = archive.tell()
                        archive.write(struct.pack("<Q", len(encoded.container)))
                        archive.write(encoded.container)
                        rows.append(
                            {
                                "run_id": run_id,
                                "state_id": state.state_id,
                                "layer_idx": state.layer_idx,
                                "layer_type": state.layer_type,
                                "state_kind": state.state_kind,
                                "state_idx": state.state_idx,
                                "dtype": state.dtype,
                                "element_size": state.element_size,
                                "block_size": block_size,
                                "block_index": block_idx,
                                "tensor_byte_offset": offset,
                                "raw_bytes": len(raw),
                                "codec": codec,
                                "payload_bytes": encoded.payload_bytes,
                                "probability_table_bytes": encoded.table_bytes,
                                "header_bytes": encoded.header_bytes + 8,
                                "required_padding_bytes": encoded.padding_bytes,
                                "compressed_total_bytes": encoded.total_bytes + 8,
                                "raw_sha256": sha256_bytes(raw),
                                "decoded_sha256": sha256_bytes(encoded.decoded),
                                "bit_exact": "true",
                                "artifact_path": str(archive_path),
                                "artifact_frame_offset": frame_offset,
                                "artifact_container_bytes": len(encoded.container),
                            }
                        )
                for state in states:
                    decoded_bytes = bytes(decoded_by_state[state.state_id])
                    if decoded_bytes != state.raw:
                        raise AssertionError(f"full state reconstruction failed: {state.state_id}")
                    if (codec, block_size) == verification_choice:
                        decoded_for_rebuild[state.state_id] = decoded_bytes
    return rows, decoded_for_rebuild, verification_choice


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def run_stage(
    *,
    cfg: dict[str, Any],
    root: Path,
    stage: str,
    run_id: str,
    codecs: list[str],
    block_sizes: list[int],
    command: list[str],
) -> Path:
    result_dir = root / "results" / run_id
    artifact_dir = root / cfg["runtime"]["raw_artifact_root"] / run_id
    if result_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"run_id already exists: {run_id}")
    result_dir.mkdir(parents=True)
    _write_json(result_dir / "env_summary.json", environment_summary(cfg, command))
    sample, input_ids_cpu = load_prepared_sample(cfg, root, stage)
    _append_jsonl(result_dir / "manifest.jsonl", [{"run_id": run_id, **sample}])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to claim a real model forward")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.monotonic()
    model = Qwen3_5ForCausalLM.from_pretrained(
        cfg["model"]["id"],
        revision=cfg["model"]["revision"],
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    load_seconds = time.monotonic() - load_started
    input_ids = input_ids_cpu.to(device)
    forward_started = time.monotonic()
    with torch.inference_mode():
        output = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize(device)
    forward_seconds = time.monotonic() - forward_started
    cache = output.past_key_values
    layer_types = list(model.config.get_text_config().layer_types)
    states = extract_persistent_states(cache, layer_types, input_ids.shape[1])
    counts = Counter(state.state_kind for state in states)
    layer_counts = Counter(layer_types)
    raw_by_kind: dict[str, int] = defaultdict(int)
    for state in states:
        raw_by_kind[state.state_kind] += len(state.raw)
    audit = {
        "run_id": run_id,
        "stage": stage,
        "model_id": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "input_tokens": input_ids.shape[1],
        "layer_type_counts": dict(layer_counts),
        "state_kind_counts": dict(counts),
        "raw_bytes_by_kind": dict(raw_by_kind),
        "kv_only_raw_bytes": raw_by_kind["key"] + raw_by_kind["value"],
        "complete_persistent_state_raw_bytes": sum(raw_by_kind.values()),
        "model_load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "attention_implementation": model.config.get_text_config()._attn_implementation,
        "persistent_state_dtypes": dict(Counter(state.dtype for state in states)),
    }
    _write_json(result_dir / "cache_audit.json", audit)
    _append_jsonl(result_dir / "states.jsonl", [state.metadata() for state in states])

    expected = {
        "full_attention": 8,
        "linear_attention": 24,
        "key": 8,
        "value": 8,
        "recurrent": 24,
        "convolution": 24,
    }
    actual = {**layer_counts, **counts}
    mismatches = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        _write_json(result_dir / "audit_mismatch.json", mismatches)
        raise AssertionError(f"runtime cache structure differs from audited config: {mismatches}")

    verification: dict[str, Any] = {
        "inventory_matches_config": True,
        "all_state_checksums_recorded": True,
        "codec_run": bool(codecs),
    }
    if codecs:
        rows, decoded, verification_choice = encode_states(
            states=states,
            codecs=codecs,
            block_sizes=block_sizes,
            zstd_level=int(cfg["codec"]["zstd_level"]),
            artifact_dir=artifact_dir,
            run_id=run_id,
        )
        _write_blocks_csv(result_dir / "blocks.csv", rows)
        summaries = summarize_blocks(rows)
        _write_summary_csv(result_dir / "summary.csv", summaries)
        decoded_cache = rebuild_cache(model.config, states, decoded, device)
        decoded_states = extract_persistent_states(decoded_cache, layer_types, input_ids.shape[1])
        decoded_by_id = {state.state_id: state.raw for state in decoded_states}
        rehydrated_exact = all(decoded_by_id[state.state_id] == state.raw for state in states)
        if not rehydrated_exact:
            raise AssertionError("rehydrated DynamicCache differs bytewise")
        next_token = input_ids[:, -1:]
        with torch.inference_mode():
            control = model.model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state.detach().cpu()
            restored = model.model(
                input_ids=next_token,
                past_key_values=decoded_cache,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state.detach().cpu()
        exact_hidden = torch.equal(control, restored)
        max_abs = float((control.float() - restored.float()).abs().max())
        verification.update(
            {
                "codec_configuration_for_cache_rebuild": {
                    "codec": verification_choice[0],
                    "block_size": verification_choice[1],
                },
                "blocks": len(rows),
                "bit_exact_blocks": sum(row["bit_exact"] == "true" for row in rows),
                "all_blocks_bit_exact": all(row["bit_exact"] == "true" for row in rows),
                "rehydrated_cache_byte_exact": rehydrated_exact,
                "continuation_hidden_exact": exact_hidden,
                "continuation_hidden_max_abs_diff": max_abs,
                "continuation_hidden_allclose_rtol_1e-5_atol_1e-5": bool(
                    torch.allclose(control.float(), restored.float(), rtol=1e-5, atol=1e-5)
                ),
            }
        )
    _write_json(result_dir / "verification.json", verification)
    del output, cache, model
    torch.cuda.empty_cache()
    return result_dir
