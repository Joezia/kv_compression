from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

import constriction
import numpy as np
import zstandard


MAGIC = b"AKV1"
PREFIX = struct.Struct("<4sI")
TABLE_DTYPE = np.dtype("<u4")


@dataclass(frozen=True)
class EncodedBlock:
    container: bytes
    payload_bytes: int
    table_bytes: int
    header_bytes: int
    padding_bytes: int
    decoded: bytes

    @property
    def total_bytes(self) -> int:
        return len(self.container)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ans_encode_stream(data: bytes) -> tuple[bytes, np.ndarray]:
    symbols = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    weights = np.bincount(symbols, minlength=256).astype(np.uint32) + 1
    model = constriction.stream.model.Categorical(
        weights.astype(np.float64), perfect=False
    )
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model)
    words = coder.get_compressed().astype(TABLE_DTYPE, copy=False)
    return words.tobytes(), weights.astype(TABLE_DTYPE, copy=False)


def _ans_decode_stream(payload: bytes, weights: np.ndarray, n: int) -> bytes:
    words = np.frombuffer(payload, dtype=TABLE_DTYPE).astype(np.uint32, copy=True)
    model = constriction.stream.model.Categorical(
        weights.astype(np.float64), perfect=False
    )
    coder = constriction.stream.stack.AnsCoder(words)
    decoded = coder.decode(model, n).astype(np.uint8, copy=False)
    return decoded.tobytes()


def _pack(
    header: dict[str, Any], tables: list[np.ndarray], payloads: list[bytes]
) -> tuple[bytes, int, int]:
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    table_blob = b"".join(t.astype(TABLE_DTYPE, copy=False).tobytes() for t in tables)
    container = PREFIX.pack(MAGIC, len(header_bytes)) + header_bytes + table_blob + b"".join(payloads)
    return container, PREFIX.size + len(header_bytes), len(table_blob)


def _unpack(container: bytes) -> tuple[dict[str, Any], memoryview, int]:
    if len(container) < PREFIX.size:
        raise ValueError("container is shorter than its prefix")
    magic, header_len = PREFIX.unpack_from(container)
    if magic != MAGIC:
        raise ValueError("invalid AKV container magic")
    header_end = PREFIX.size + header_len
    if header_end > len(container):
        raise ValueError("truncated AKV header")
    header = json.loads(container[PREFIX.size:header_end])
    return header, memoryview(container), header_end


def encode_block(
    raw: bytes,
    *,
    codec: str,
    element_size: int,
    metadata: dict[str, Any],
    zstd_level: int = 3,
) -> EncodedBlock:
    if element_size <= 0:
        raise ValueError("element_size must be positive")
    if codec in {"ans_raw", "ans_byte_lane"}:
        lane_count = 1 if codec == "ans_raw" else element_size
        streams = [raw] if lane_count == 1 else [raw[lane::lane_count] for lane in range(lane_count)]
        payloads: list[bytes] = []
        tables: list[np.ndarray] = []
        stream_headers: list[dict[str, int]] = []
        for lane, stream in enumerate(streams):
            payload, table = _ans_encode_stream(stream)
            payloads.append(payload)
            tables.append(table)
            stream_headers.append(
                {"lane": lane, "symbols": len(stream), "payload_bytes": len(payload)}
            )
        header = {
            "version": 1,
            "codec": codec,
            "raw_bytes": len(raw),
            "element_size": element_size,
            "table_entries": 256,
            "table_entry_bytes": 4,
            "streams": stream_headers,
            "metadata": metadata,
        }
        container, header_bytes, table_bytes = _pack(header, tables, payloads)
    elif codec == "zstd":
        payload = zstandard.ZstdCompressor(level=zstd_level).compress(raw)
        payloads = [payload]
        header = {
            "version": 1,
            "codec": codec,
            "raw_bytes": len(raw),
            "element_size": element_size,
            "zstd_level": zstd_level,
            "streams": [{"lane": 0, "symbols": len(raw), "payload_bytes": len(payload)}],
            "metadata": metadata,
        }
        container, header_bytes, table_bytes = _pack(header, [], payloads)
    else:
        raise ValueError(f"unsupported codec: {codec}")

    decoded = decode_block(container)
    return EncodedBlock(
        container=container,
        payload_bytes=sum(len(p) for p in payloads),
        table_bytes=table_bytes,
        header_bytes=header_bytes,
        padding_bytes=0,
        decoded=decoded,
    )


def decode_block(container: bytes) -> bytes:
    header, view, cursor = _unpack(container)
    codec = header["codec"]
    streams = header["streams"]
    decoded_streams: list[bytes] = []
    if codec in {"ans_raw", "ans_byte_lane"}:
        tables: list[np.ndarray] = []
        for _ in streams:
            table_end = cursor + 256 * 4
            if table_end > len(view):
                raise ValueError("truncated ANS probability table")
            tables.append(np.frombuffer(view[cursor:table_end], dtype=TABLE_DTYPE).copy())
            cursor = table_end
        for stream, table in zip(streams, tables, strict=True):
            payload_end = cursor + int(stream["payload_bytes"])
            if payload_end > len(view):
                raise ValueError("truncated ANS payload")
            decoded_streams.append(
                _ans_decode_stream(
                    bytes(view[cursor:payload_end]), table, int(stream["symbols"])
                )
            )
            cursor = payload_end
        if codec == "ans_raw":
            decoded = decoded_streams[0]
        else:
            raw_len = int(header["raw_bytes"])
            lane_count = len(decoded_streams)
            merged = bytearray(raw_len)
            for lane, stream in enumerate(decoded_streams):
                merged[lane::lane_count] = stream
            decoded = bytes(merged)
    elif codec == "zstd":
        stream = streams[0]
        payload_end = cursor + int(stream["payload_bytes"])
        if payload_end > len(view):
            raise ValueError("truncated zstd payload")
        decoded = zstandard.ZstdDecompressor().decompress(
            bytes(view[cursor:payload_end]), max_output_size=int(header["raw_bytes"])
        )
        cursor = payload_end
    else:
        raise ValueError(f"unsupported codec in container: {codec}")

    if cursor != len(view):
        raise ValueError("unexpected trailing bytes in AKV container")
    if len(decoded) != int(header["raw_bytes"]):
        raise ValueError("decoded length does not match header")
    return decoded


# ---------------------------------------------------------------------------
# AKV2: compact container for the per-turn (agentic) experiments.
#
# Differences from AKV1, all deliberate and accounted for:
# - binary header (no JSON / debug metadata inside the container; the block
#   index lives in blocks/units files and a real store keeps its own index);
# - per-block probability tables are quantized to uint16 counts summing to
#   2**TABLE_PRECISION_BITS, so the decoder rebuilds the identical model;
# - "shared" codecs reference an externally supplied table set (built offline
#   from a held-out calibration session). Those tables are NOT inside the
#   container; their size is reported once per table set by the caller.
# ---------------------------------------------------------------------------

MAGIC2 = b"AKV2"
HEADER2 = struct.Struct("<4sBBBI")  # magic, codec id, lanes, table mode, raw_len
STREAM2 = struct.Struct("<II")  # symbols, payload_bytes
TABLE16_DTYPE = np.dtype("<u2")
TABLE_PRECISION_BITS = 15
CODEC2_IDS = {"zstd": 1, "ans_raw": 2, "ans_lane": 3, "ans_lane_shared": 4}
CODEC2_NAMES = {value: key for key, value in CODEC2_IDS.items()}
TABLE_MODE_NONE, TABLE_MODE_INLINE, TABLE_MODE_SHARED = 0, 1, 2


def quantize_counts(counts: np.ndarray, precision_bits: int = TABLE_PRECISION_BITS) -> np.ndarray:
    """Map a 256-entry histogram to positive integer weights summing to 2**bits.

    Every symbol keeps weight >= 1 so any byte stays encodable (needed when a
    shared table meets data it was not calibrated on).
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (256,):
        raise ValueError("expected a 256-entry histogram")
    total = 1 << precision_bits
    free = total - 256
    s = counts.sum()
    if s <= 0:
        weights = np.ones(256, dtype=np.int64)
        weights[0] += free
    else:
        extra = np.floor(counts / s * free).astype(np.int64)
        remainder = free - int(extra.sum())
        if remainder:
            order = np.argsort(-(counts / s * free - extra), kind="stable")
            extra[order[:remainder]] += 1
        weights = extra + 1
    assert int(weights.sum()) == total and int(weights.min()) >= 1
    return weights.astype(np.uint16)


def lane_streams(raw: bytes, lanes: int) -> list[bytes]:
    return [raw] if lanes == 1 else [raw[lane::lanes] for lane in range(lanes)]


def lane_histograms(raw: bytes, lanes: int) -> np.ndarray:
    """Byte histograms per lane, shape [lanes, 256]."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    return np.stack(
        [np.bincount(arr[lane::lanes], minlength=256) for lane in range(lanes)]
    ).astype(np.int64)


def order0_entropy_bytes(raw: bytes, lanes: int) -> float:
    """Empirical order-0 entropy (bytes) of the lane-split data. Diagnostic only."""
    total = 0.0
    for hist in lane_histograms(raw, lanes):
        n = hist.sum()
        if n == 0:
            continue
        p = hist[hist > 0] / n
        total += float(-(hist[hist > 0] * np.log2(p)).sum()) / 8.0
    return total


def _ans2_encode(stream: bytes, weights: np.ndarray) -> bytes:
    if not stream:
        return b""
    symbols = np.frombuffer(stream, dtype=np.uint8).astype(np.int32)
    model = constriction.stream.model.Categorical(
        weights.astype(np.float64), perfect=False
    )
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model)
    return coder.get_compressed().astype(TABLE_DTYPE, copy=False).tobytes()


def _ans2_decode(payload: bytes, weights: np.ndarray, n: int) -> bytes:
    if n == 0:
        return b""
    words = np.frombuffer(payload, dtype=TABLE_DTYPE).astype(np.uint32, copy=True)
    model = constriction.stream.model.Categorical(
        weights.astype(np.float64), perfect=False
    )
    return constriction.stream.stack.AnsCoder(words).decode(model, n).astype(np.uint8).tobytes()


@dataclass(frozen=True)
class Encoded2:
    container: bytes
    payload_bytes: int
    table_bytes: int
    header_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.container)


def encode2(
    raw: bytes,
    *,
    codec: str,
    element_size: int,
    shared_tables: np.ndarray | None = None,
    zstd_level: int = 3,
) -> Encoded2:
    """Encode one unit. `shared_tables` is [lanes, 256] uint16 for ans_lane_shared."""
    if codec not in CODEC2_IDS:
        raise ValueError(f"unsupported codec: {codec}")
    if len(raw) >= 1 << 32:
        raise ValueError("unit too large for AKV2")
    if codec == "zstd":
        payload = zstandard.ZstdCompressor(level=zstd_level).compress(raw)
        head = HEADER2.pack(MAGIC2, CODEC2_IDS[codec], 1, TABLE_MODE_NONE, len(raw))
        streams = STREAM2.pack(len(raw), len(payload))
        container = head + streams + payload
        return Encoded2(container, len(payload), 0, len(head) + len(streams))

    lanes = 1 if codec == "ans_raw" else element_size
    if len(raw) % lanes:
        raise ValueError("raw length is not a multiple of the lane count")
    parts = lane_streams(raw, lanes)
    if codec == "ans_lane_shared":
        if shared_tables is None or shared_tables.shape != (lanes, 256):
            raise ValueError(f"ans_lane_shared needs shared tables of shape ({lanes}, 256)")
        tables = shared_tables.astype(np.uint16)
        mode = TABLE_MODE_SHARED
    else:
        tables = np.stack([quantize_counts(h) for h in lane_histograms(raw, lanes)])
        mode = TABLE_MODE_INLINE
    payloads = [_ans2_encode(part, table) for part, table in zip(parts, tables, strict=True)]
    head = HEADER2.pack(MAGIC2, CODEC2_IDS[codec], lanes, mode, len(raw))
    stream_heads = b"".join(STREAM2.pack(len(p), len(c)) for p, c in zip(parts, payloads, strict=True))
    table_blob = tables.astype(TABLE16_DTYPE).tobytes() if mode == TABLE_MODE_INLINE else b""
    container = head + stream_heads + table_blob + b"".join(payloads)
    return Encoded2(
        container,
        sum(len(p) for p in payloads),
        len(table_blob),
        len(head) + len(stream_heads),
    )


def decode2(container: bytes, *, element_size: int, shared_tables: np.ndarray | None = None) -> bytes:
    if len(container) < HEADER2.size:
        raise ValueError("AKV2 container too short")
    magic, codec_id, lanes, mode, raw_len = HEADER2.unpack_from(container)
    if magic != MAGIC2:
        raise ValueError("invalid AKV2 magic")
    codec = CODEC2_NAMES.get(codec_id)
    if codec is None:
        raise ValueError(f"unknown AKV2 codec id {codec_id}")
    cursor = HEADER2.size
    streams = []
    for _ in range(lanes):
        if cursor + STREAM2.size > len(container):
            raise ValueError("truncated AKV2 stream header")
        streams.append(STREAM2.unpack_from(container, cursor))
        cursor += STREAM2.size
    if codec == "zstd":
        symbols, plen = streams[0]
        payload = container[cursor : cursor + plen]
        if len(payload) != plen:
            raise ValueError("truncated zstd payload")
        cursor += plen
        decoded = zstandard.ZstdDecompressor().decompress(payload, max_output_size=raw_len) if raw_len else b""
    else:
        expected_lanes = 1 if codec == "ans_raw" else element_size
        if lanes != expected_lanes:
            raise ValueError("lane count does not match element size")
        if mode == TABLE_MODE_INLINE:
            end = cursor + lanes * 256 * 2
            if end > len(container):
                raise ValueError("truncated AKV2 table")
            tables = np.frombuffer(container[cursor:end], dtype=TABLE16_DTYPE).reshape(lanes, 256)
            cursor = end
        elif mode == TABLE_MODE_SHARED:
            if shared_tables is None or shared_tables.shape != (lanes, 256):
                raise ValueError("decoder needs the shared table set")
            tables = shared_tables
        else:
            raise ValueError("invalid table mode for ANS")
        parts = []
        for (symbols, plen), table in zip(streams, tables, strict=True):
            payload = container[cursor : cursor + plen]
            if len(payload) != plen:
                raise ValueError("truncated ANS payload")
            cursor += plen
            parts.append(_ans2_decode(payload, table, symbols))
        if lanes == 1:
            decoded = parts[0]
        else:
            merged = bytearray(raw_len)
            for lane, part in enumerate(parts):
                merged[lane::lanes] = part
            decoded = bytes(merged)
    if cursor != len(container):
        raise ValueError("unexpected trailing bytes in AKV2 container")
    if len(decoded) != raw_len:
        raise ValueError("decoded length does not match header")
    return decoded
