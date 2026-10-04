from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kv_codec import decode_block, encode_block  # noqa: E402


@pytest.mark.parametrize("codec", ["ans_raw", "ans_byte_lane", "zstd"])
@pytest.mark.parametrize("element_size", [1, 2, 4])
@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\x00",
        bytes(range(256)),
        b"\x00\x80\x7f\xff" * 257,
    ],
)
def test_roundtrip(codec: str, element_size: int, raw: bytes) -> None:
    encoded = encode_block(
        raw,
        codec=codec,
        element_size=element_size,
        metadata={"test": True},
    )
    assert encoded.decoded == raw
    assert decode_block(encoded.container) == raw
    assert encoded.total_bytes == len(encoded.container)


def test_truncated_container_is_rejected() -> None:
    encoded = encode_block(
        bytes(range(64)), codec="ans_raw", element_size=1, metadata={"test": True}
    )
    with pytest.raises(ValueError):
        decode_block(encoded.container[:-1])
