"""Run the complete five-inner-fold SREA-MIL evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srea_mil.engine import run_protocol
from srea_mil.model import MODEL_SPECS, SREAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/paper_protocol.json")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="srea_mil")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = SREAConfig(**config.pop("model"))
    summary = run_protocol(
        manifest_path=args.manifest,
        splits_path=args.splits,
        output_dir=args.output,
        model_name=args.model,
        cfg=model_config,
        protocol=config,
        device=args.device,
    )
    print(json.dumps(summary["held_out_metrics"], indent=2))


if __name__ == "__main__":
    main()
