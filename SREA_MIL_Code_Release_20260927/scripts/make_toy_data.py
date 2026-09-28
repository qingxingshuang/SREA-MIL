"""Create synthetic features and five-fold splits for an installation check."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="toy_data")
    parser.add_argument("--slides", type=int, default=30)
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.slides < 20:
        raise ValueError("use at least 20 slides for the five-fold toy split")
    root = Path(args.output).resolve()
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[list[object]] = []
    identifiers: list[str] = []
    for index in range(args.slides):
        slide_id = f"toy_{index:03d}"
        label = index % 2
        identifiers.append(slide_id)
        abnormal = rng.normal(label * 0.15, 1.0, (args.instances, 1280)).astype("float32")
        normal = rng.normal(-label * 0.05, 1.0, (args.instances, 1280)).astype("float32")
        abnormal_path = feature_dir / f"{slide_id}_abnormal.npy"
        normal_path = feature_dir / f"{slide_id}_normal.npy"
        np.save(abnormal_path, abnormal)
        np.save(normal_path, normal)
        rows.append(
            [
                slide_id,
                label,
                abnormal_path.relative_to(root).as_posix(),
                normal_path.relative_to(root).as_posix(),
            ]
        )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["slide_id", "label", "abnormal_feature_path", "normal_feature_path"]
        )
        writer.writerows(rows)
    development = identifiers[:20]
    held_out = identifiers[20:]
    folds = []
    for fold_index in range(5):
        validation = development[fold_index::5]
        training = [slide_id for slide_id in development if slide_id not in validation]
        folds.append({"train": training, "val": validation})
    splits = {
        "development": development,
        "held_out": held_out,
        "inner_folds": folds,
    }
    (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
