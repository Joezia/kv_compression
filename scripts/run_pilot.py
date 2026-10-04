#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pilot import load_config, prepare_agentx_samples, run_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 AgentX persistent-cache pilot")
    parser.add_argument("command", choices=["prepare", "m0", "m1", "m2a", "m2b", "m2c"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--codecs", nargs="+", choices=["ans_raw", "ans_byte_lane", "zstd"])
    parser.add_argument("--block-sizes", nargs="+", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        for record in prepare_agentx_samples(cfg, ROOT):
            print(record["stage"], record["trace_id"], record["request_path"], record["used_input_tokens"], record["input_ids_sha256"])
        return
    if not args.run_id:
        parser.error("--run-id is required for experiment stages")
    default_codecs = [] if args.command == "m0" else ["ans_raw"]
    default_blocks = [262144]
    result = run_stage(
        cfg=cfg,
        root=ROOT,
        stage=args.command,
        run_id=args.run_id,
        codecs=args.codecs if args.codecs is not None else default_codecs,
        block_sizes=args.block_sizes if args.block_sizes is not None else default_blocks,
        command=sys.argv,
    )
    print(result)


if __name__ == "__main__":
    main()
