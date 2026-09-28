"""Feature-manifest dataset for the two-stream cervical WSI protocol."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


LABEL_MAP = {
    "0": 0,
    "nilm": 0,
    "normal": 0,
    "negative": 0,
    "1": 1,
    "abnormal": 1,
    "positive": 1,
}


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    label: int
    abnormal_feature_path: Path | None
    normal_feature_path: Path | None


def _parse_label(value: str) -> int:
    key = value.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"unsupported label {value!r}; use 0/1 or NILM/abnormal")
    return LABEL_MAP[key]


def _resolve_feature_path(root: Path, value: str) -> Path | None:
    value = value.strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def read_manifest(path: str | Path) -> list[SlideRecord]:
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    required = {
        "slide_id",
        "label",
        "abnormal_feature_path",
        "normal_feature_path",
    }
    records: list[SlideRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for row in reader:
            slide_id = row["slide_id"].strip()
            if not slide_id:
                raise ValueError("slide_id cannot be empty")
            record = SlideRecord(
                slide_id=slide_id,
                label=_parse_label(row["label"]),
                abnormal_feature_path=_resolve_feature_path(
                    root, row["abnormal_feature_path"]
                ),
                normal_feature_path=_resolve_feature_path(
                    root, row["normal_feature_path"]
                ),
            )
            if record.abnormal_feature_path is None and record.normal_feature_path is None:
                raise ValueError(f"slide {slide_id!r} has no feature stream")
            records.append(record)
    if len({record.slide_id for record in records}) != len(records):
        raise ValueError("slide_id values must be unique")
    return records


def _load_feature(path: Path | None, input_dim: int) -> Tensor:
    if path is None:
        return torch.empty((0, input_dim), dtype=torch.float32)
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != input_dim:
        raise ValueError(
            f"{path} must have shape [instances, {input_dim}], got {array.shape}"
        )
    if array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{path} must contain finite, nonempty features")
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _deterministic_cap(
    features: Tensor,
    max_instances: int,
    seed: int,
    slide_id: str,
    stream: str,
    epoch: int,
) -> Tensor:
    if features.shape[0] <= max_instances:
        return features
    token = f"{seed}|{slide_id}|{stream}|{epoch}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    indices = np.sort(rng.choice(features.shape[0], max_instances, replace=False))
    return features[torch.from_numpy(indices).long()]


class FeatureManifestDataset(Dataset[dict[str, object]]):
    """Load precomputed normal and suspected-abnormal candidate features."""

    def __init__(
        self,
        manifest: str | Path | Sequence[SlideRecord],
        slide_ids: Iterable[str] | None = None,
        input_dim: int = 1280,
        max_instances: int = 512,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
        if slide_ids is not None:
            wanted = set(slide_ids)
            records = [record for record in records if record.slide_id in wanted]
            found = {record.slide_id for record in records}
            missing = wanted.difference(found)
            if missing:
                raise ValueError(f"slide IDs absent from manifest: {sorted(missing)[:5]}")
        self.records = records
        self.input_dim = input_dim
        self.max_instances = max_instances
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        abnormal = _load_feature(record.abnormal_feature_path, self.input_dim)
        normal = _load_feature(record.normal_feature_path, self.input_dim)
        abnormal = _deterministic_cap(
            abnormal, self.max_instances, self.seed, record.slide_id, "abnormal", self.epoch
        )
        normal = _deterministic_cap(
            normal, self.max_instances, self.seed, record.slide_id, "normal", self.epoch
        )
        return {
            "slide_id": record.slide_id,
            "label": record.label,
            "abnormal": abnormal,
            "normal": normal,
        }


def _pad_stream(items: Sequence[Tensor], input_dim: int) -> tuple[Tensor, Tensor]:
    max_length = max((item.shape[0] for item in items), default=0)
    max_length = max(max_length, 1)
    batch = torch.zeros((len(items), max_length, input_dim), dtype=torch.float32)
    mask = torch.zeros((len(items), max_length), dtype=torch.bool)
    for index, item in enumerate(items):
        length = item.shape[0]
        if length:
            batch[index, :length] = item
            mask[index, :length] = True
    return batch, mask


def collate_slides(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    abnormal_items = [sample["abnormal"] for sample in samples]
    normal_items = [sample["normal"] for sample in samples]
    assert all(isinstance(item, Tensor) for item in abnormal_items + normal_items)
    input_dim = int((abnormal_items + normal_items)[0].shape[1])
    abnormal, abnormal_mask = _pad_stream(abnormal_items, input_dim)
    normal, normal_mask = _pad_stream(normal_items, input_dim)
    return {
        "slide_id": [str(sample["slide_id"]) for sample in samples],
        "label": torch.tensor([int(sample["label"]) for sample in samples], dtype=torch.long),
        "abnormal": abnormal,
        "abnormal_mask": abnormal_mask,
        "normal": normal,
        "normal_mask": normal_mask,
    }
