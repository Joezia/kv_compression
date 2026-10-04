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
