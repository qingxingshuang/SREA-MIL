"""Evaluate a saved refit checkpoint on a manifest subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srea_mil.data import FeatureManifestDataset, collate_slides
from srea_mil.engine import predict
from srea_mil.metrics import binary_metrics
from srea_mil.model import SREAConfig, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--slide-ids", required=True, help="JSON list of slide IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-instances", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SREAConfig(**checkpoint["model_config"])
    model = build_model(checkpoint["model_name"], config)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    model.to(device)
    slide_ids = json.loads(Path(args.slide_ids).read_text(encoding="utf-8"))
    dataset = FeatureManifestDataset(
        args.manifest,
        slide_ids=slide_ids,
        input_dim=config.input_dim,
        max_instances=args.max_instances,
        seed=args.seed,
        epoch=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_slides,
    )
    identifiers, labels, probabilities = predict(model, loader, device)
    metrics = binary_metrics(labels, probabilities)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "label", "abnormal_probability"])
        writer.writerows(zip(identifiers, labels.tolist(), probabilities.tolist()))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
