"""Turn-ordered inputs for the per-turn (agentic) experiments.

Two sources:
- AgentX/Weka traces: real session structure (lengths, prefix relations via
  hash_ids, timing), content synthesized by AIPerf for the target tokenizer.
- Message trajectories (e.g. SWE-agent/OpenHands): real content, rendered with
  the target model's chat template. Prefix relations are measured on the
  resulting token IDs, never assumed.

A "turn" here is one model request of the main thread. Subagent requests are
counted but not replayed in this round.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MAIN_REQUEST_TYPES = {"n", "s"}


@dataclass
class Turn:
    turn_idx: int
    request_path: str
    token_ids: np.ndarray  # uint32
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    source_kind: str
    turns: list[Turn]
    meta: dict[str, Any] = field(default_factory=dict)


def drop_arrow_union_nulls(value: Any) -> Any:
    """Remove Arrow union padding (null-valued keys) before strict validation."""
    if isinstance(value, dict):
        return {k: drop_arrow_union_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_arrow_union_nulls(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Prefix-chain block keys (shared by trace analysis and the GPU runner)
# --------------------------------------------------------------------------


def chain_keys(blocks: Iterable[bytes]) -> list[bytes]:
    """key_i = H(key_{i-1} || block_i). Equal key <=> equal full prefix.

    A KV block can only be reused when its whole prefix matches, so this is
    the correct identity for prefix-cache deduplication.
    """
    keys: list[bytes] = []
    prev = b""
    for block in blocks:
        prev = hashlib.blake2b(prev + block, digest_size=16).digest()
        keys.append(prev)
    return keys


def token_block_keys(token_ids: np.ndarray, block_tokens: int) -> list[bytes]:
    ids = np.asarray(token_ids, dtype="<u4")
    full = len(ids) // block_tokens
    return chain_keys(ids[i * block_tokens : (i + 1) * block_tokens].tobytes() for i in range(full))


def hash_id_block_keys(hash_ids: list[int]) -> list[bytes]:
    return chain_keys(int(h).to_bytes(8, "little", signed=True) for h in hash_ids)


def common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(np.asarray(a[:n]) != np.asarray(b[:n]))[0]
    return int(diff[0]) if len(diff) else n


# --------------------------------------------------------------------------
# AgentX / Weka traces
# --------------------------------------------------------------------------


def load_trace_rows(dataset_id: str, revision: str, split: str, indices: list[int], cache_dir: Path) -> dict[int, dict]:
    """Return raw trace rows by index, caching each row as JSON (same layout as the pilot)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict] = {}
    missing = [i for i in indices if not (cache_dir / f"agentx_trace_{i}.json").exists()]
    if missing:
        from datasets import load_dataset

        dataset = load_dataset(dataset_id, split=split, revision=revision, streaming=True)
        for index, row in enumerate(dataset):
            if index in missing:
                (cache_dir / f"agentx_trace_{index}.json").write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
            if index >= max(missing):
                break
    for i in indices:
        rows[i] = json.loads((cache_dir / f"agentx_trace_{i}.json").read_text(encoding="utf-8"))
    return rows


def iter_trace_rows(dataset_id: str, revision: str, split: str, limit: int) -> Iterable[tuple[int, dict]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, revision=revision, streaming=True)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield index, row


def _field(req: dict, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in req and req[name] is not None:
            return req[name]
    return default


def trace_turn_stats(row: dict, cache_block_tokens: int | None = None) -> tuple[list[dict], dict]:
    """Hash-level per-turn statistics of one raw trace row (no tokenizer, no GPU).

    Dedup uses prefix-chain keys over hash_ids; tokens not covered by hash_ids
    (un-hashed tail) are always counted as new.
    """
    row = drop_arrow_union_nulls(row)
    block = int(row["block_size"])
    if cache_block_tokens not in (None, block):
        raise ValueError("hash-level analysis can only use the trace block size")
    seen: set[bytes] = set()
    prev_keys: list[bytes] = []
    prev_t: float | None = None
    turns: list[dict] = []
    sub_requests = sub_input_tokens = 0
    main_idx = 0
    for outer_idx, req in enumerate(row["requests"]):
        if req.get("type") == "subagent":
            for inner in req.get("requests", []) or []:
                sub_requests += 1
                sub_input_tokens += int(_field(inner, "in", "input_length", default=0))
            continue
        if req.get("type") not in MAIN_REQUEST_TYPES:
            continue
        hash_ids = list(req.get("hash_ids") or [])
        n_in = int(_field(req, "in", "input_length"))
        keys = hash_id_block_keys(hash_ids)
        # A partial final hash block is never a reusable full cache block.
        full = min(len(keys), n_in // block)
        keys = keys[:full]
        lcp_prev = 0
        for a, b in zip(keys, prev_keys):
            if a != b:
                break
            lcp_prev += 1
        hit = 0
        for key in keys:
            if key in seen:
                hit += 1
            else:
                break
        seen.update(keys)
        t = float(req.get("t", 0.0))
        turns.append(
            {
                "turn_idx": main_idx,
                "request_path": f"main/{outer_idx}",
                "t": t,
                "gap_s": None if prev_t is None else t - prev_t,
                "think_time": req.get("think_time"),
                "api_time": req.get("api_time"),
                "input_tokens": n_in,
                "output_tokens": int(_field(req, "out", "output_length", default=0)),
                "hash_blocks": len(hash_ids),
                "full_blocks": full,
                "lcp_prev_blocks": lcp_prev,
                "hit_blocks": hit,
                "new_blocks": full - hit,
                "uncovered_tail_tokens": n_in - full * block,
                "stop": req.get("stop"),
            }
        )
        prev_keys = keys
        prev_t = t
        main_idx += 1
    summary = {
        "session_id": row["id"],
        "block_size": block,
        "main_turns": len(turns),
        "subagent_requests": sub_requests,
        "subagent_input_tokens": sub_input_tokens,
        "snapshot_tokens": sum(t["input_tokens"] for t in turns),
        "unique_block_tokens": len(seen) * block,
        "new_tokens_incl_tail": sum(t["new_blocks"] * block + t["uncovered_tail_tokens"] for t in turns),
        "max_input_tokens": max((t["input_tokens"] for t in turns), default=0),
        "duration_s": (turns[-1]["t"] - turns[0]["t"]) if turns else 0.0,
    }
    summary["dedup_ratio_tokens"] = (
        summary["snapshot_tokens"] / summary["new_tokens_incl_tail"] if summary["new_tokens_incl_tail"] else None
    )
    return turns, summary


def agentx_session(
    row: dict,
    *,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    corpus: str,
    seed: int,
    max_turns: int | None = None,
) -> Session:
    """Synthesize every main-thread request of one trace with AIPerf, in order."""
    from aiperf.common import random_generator as aiperf_rng
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.dataset.content import PromptConfig
    from aiperf.dataset.generator.corpus import resolve_prompt_generator
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    aiperf_rng.init(int(seed))
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision, resolve_alias=False)
    generator = resolve_prompt_generator(
        corpus=corpus, default_corpus=corpus, tokenizer=tokenizer, prompts=PromptConfig()
    )
    trace = WekaTrace.model_validate(drop_arrow_union_nulls(row))
    generator._cache.clear()
    generator._hash_id_corpus_rng.set_trace_id(trace.id)
    turns: list[Turn] = []
    skipped_subagents = 0
    for outer_idx, req in enumerate(trace.requests):
        if getattr(req, "type", None) == "subagent":
            skipped_subagents += 1
            continue
        ids = generator._build_token_sequence(req.input_length, req.hash_ids, trace.block_size)
        if len(ids) != req.input_length:
            raise AssertionError("AIPerf synthesis did not reproduce the recorded input length")
        turns.append(
            Turn(
                turn_idx=len(turns),
                request_path=f"main/{outer_idx}",
                token_ids=np.asarray(ids, dtype=np.uint32),
                meta={
                    "t": req.t,
                    "think_time": req.think_time,
                    "recorded_input_tokens": req.input_length,
                    "output_tokens": req.output_length,
                    "hash_blocks": len(req.hash_ids),
                    "trace_model": req.model,
                },
            )
        )
        if max_turns is not None and len(turns) >= max_turns:
            break
    return Session(
        session_id=trace.id,
        source_kind="AgentX-synth-session",
        turns=turns,
        meta={
            "trace_block_size": trace.block_size,
            "hash_id_scope": trace.hash_id_scope,
            "subagent_entries_skipped": skipped_subagents,
            "input_mapping": f"AIPerf {corpus} corpus hash-id synthesis; direct token IDs; no chat template",
        },
    )


# --------------------------------------------------------------------------
# Message trajectories (real content)
# --------------------------------------------------------------------------

_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def normalize_messages(raw_messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in raw_messages:
        role = _ROLE_MAP.get(str(_field(msg, "role", "from", default="")).lower())
        if role is None:
            raise ValueError(f"unknown message role: {msg}")
        # nebius/SWE-agent-trajectories keeps the system text under "system_prompt".
        content = _field(msg, "content", "text", "value", "system_prompt", default="")
        if isinstance(content, list):  # list-of-parts format
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        out.append({"role": role, "content": str(content)})
    return out


def trajectory_session(
    session_id: str,
    messages: list[dict],
    *,
    tokenizer,
    source_kind: str,
    max_turns: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> Session:
    """One turn per assistant message: input = all messages before it + generation prompt."""
    messages = normalize_messages(messages)
    kwargs = chat_template_kwargs or {}
    turns: list[Turn] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant" or i == 0:
            continue
        ids = tokenizer.apply_chat_template(
            messages[:i], add_generation_prompt=True, tokenize=True, **kwargs
        )
        if hasattr(ids, "keys"):  # transformers 5.x returns a BatchEncoding (a UserDict, not a dict)
            ids = ids["input_ids"]
        if len(ids) and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        turns.append(
            Turn(
                turn_idx=len(turns),
                request_path=f"msg/{i}",
                token_ids=np.asarray(ids, dtype=np.uint32),
                meta={"messages_before": i},
            )
        )
        if max_turns is not None and len(turns) >= max_turns:
            break
    return Session(session_id=session_id, source_kind=source_kind, turns=turns, meta={"messages": len(messages)})


def session_fingerprint(session: Session) -> str:
    h = hashlib.sha256()
    for turn in session.turns:
        h.update(np.asarray(turn.token_ids, dtype="<u4").tobytes())
        h.update(b"|")
    return h.hexdigest()
