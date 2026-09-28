"""Training, model selection, refitting, and held-out evaluation."""

from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import FeatureManifestDataset, SlideRecord, collate_slides, read_manifest
from .metrics import binary_metrics
from .model import SREAConfig, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def make_loader(
    records: list[SlideRecord],
    slide_ids: Iterable[str],
    cfg: SREAConfig,
    batch_size: int,
    seed: int,
    shuffle: bool,
    epoch: int,
    max_instances: int,
    num_workers: int,
) -> tuple[FeatureManifestDataset, DataLoader]:
    dataset = FeatureManifestDataset(
        records,
        slide_ids=slide_ids,
        input_dim=cfg.input_dim,
        max_instances=max_instances,
        seed=seed,
        epoch=epoch,
    )
    generator = torch.Generator().manual_seed(seed + epoch)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_slides,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_slides = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        labels = batch["label"]
        assert isinstance(labels, Tensor)
        output = model(
            batch["abnormal"],
            batch["abnormal_mask"],
            batch["normal"],
            batch["normal_mask"],
            labels,
        )
        main_loss = nn.functional.cross_entropy(output["logits"], labels)
        loss = main_loss + model.auxiliary_weight * output["auxiliary_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_slides += batch_size
    return total_loss / max(total_slides, 1)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    slide_ids: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        output = model(
            batch["abnormal"],
            batch["abnormal_mask"],
            batch["normal"],
            batch["normal_mask"],
        )
        slide_ids.extend(raw_batch["slide_id"])
        labels.extend(raw_batch["label"].tolist())
        probabilities.extend(output["probabilities"][:, 1].cpu().tolist())
    return slide_ids, np.asarray(labels), np.asarray(probabilities)


def _score_for_selection(metrics: dict[str, float | int]) -> float:
    score = float(metrics["auroc"])
    return score if math.isfinite(score) else float(metrics["accuracy"])


def select_epoch_count(
    records: list[SlideRecord],
    inner_folds: list[dict[str, list[str]]],
    model_name: str,
    cfg: SREAConfig,
    protocol: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[int, list[dict[str, Any]]]:
    fold_results: list[dict[str, Any]] = []
    selected_counts: list[int] = []
    for fold_index, fold in enumerate(inner_folds):
        fold_seed = int(protocol["seed"])
        seed_everything(fold_seed)
        model = build_model(model_name, cfg).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(protocol["learning_rate"]),
            weight_decay=float(protocol["weight_decay"]),
        )
        best_score = -float("inf")
        best_epoch = 0
        best_metrics: dict[str, float | int] = {}
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0
        history: list[dict[str, Any]] = []
        for epoch in range(int(protocol["max_epochs"])):
            train_dataset, train_loader = make_loader(
                records,
                fold["train"],
                cfg,
                int(protocol["batch_size"]),
                fold_seed,
                True,
                epoch,
                int(protocol["max_instances"]),
                int(protocol["num_workers"]),
            )
            train_dataset.set_epoch(epoch)
            train_loss = train_epoch(
                model, train_loader, optimizer, device, float(protocol["grad_clip"])
            )
            _, validation_loader = make_loader(
                records,
                fold["val"],
                cfg,
                int(protocol["batch_size"]),
                fold_seed,
                False,
                int(protocol["validation_epoch"]),
                int(protocol["max_instances"]),
                int(protocol["num_workers"]),
            )
            _, labels, probabilities = predict(model, validation_loader, device)
            metrics = binary_metrics(labels, probabilities, float(protocol["threshold"]))
            score = _score_for_selection(metrics)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= int(protocol["patience"]):
                break
        if best_state is None:
            raise RuntimeError("no checkpoint was selected")
        selected_counts.append(best_epoch)
        torch.save(
            {
                "model_name": model_name,
                "model_config": asdict(cfg),
                "state_dict": best_state,
                "selected_epoch": best_epoch,
            },
            output_dir / f"inner_fold_{fold_index + 1}.pt",
        )
        result = {
            "fold": fold_index + 1,
            "selected_epoch": best_epoch,
            "best_metrics": best_metrics,
            "history": history,
        }
        fold_results.append(result)
        (output_dir / f"inner_fold_{fold_index + 1}.json").write_text(
            json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
        )
    return int(median(selected_counts)), fold_results


def refit_and_evaluate(
    records: list[SlideRecord],
    development_ids: list[str],
    held_out_ids: list[str],
    epochs: int,
    model_name: str,
    cfg: SREAConfig,
    protocol: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(protocol["seed"])
    seed_everything(seed)
    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    refit_history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        dataset, loader = make_loader(
            records,
            development_ids,
            cfg,
            int(protocol["batch_size"]),
            seed,
            True,
            epoch,
            int(protocol["max_instances"]),
            int(protocol["num_workers"]),
        )
        dataset.set_epoch(epoch)
        loss = train_epoch(
            model, loader, optimizer, device, float(protocol["grad_clip"])
        )
        refit_history.append({"epoch": epoch + 1, "train_loss": loss})
    _, held_loader = make_loader(
        records,
        held_out_ids,
        cfg,
        int(protocol["batch_size"]),
        seed,
        False,
        int(protocol["validation_epoch"]),
        int(protocol["max_instances"]),
        int(protocol["num_workers"]),
    )
    slide_ids, labels, probabilities = predict(model, held_loader, device)
    metrics = binary_metrics(labels, probabilities, float(protocol["threshold"]))
    checkpoint_path = output_dir / "refit_model.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_config": asdict(cfg),
            "state_dict": model.state_dict(),
            "refit_epochs": epochs,
        },
        checkpoint_path,
    )
    with (output_dir / "held_out_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "label", "abnormal_probability"])
        writer.writerows(zip(slide_ids, labels.tolist(), probabilities.tolist()))
    return {
        "refit_epochs": epochs,
        "held_out_metrics": metrics,
        "refit_history": refit_history,
        "checkpoint": checkpoint_path.name,
    }


def run_protocol(
    manifest_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    cfg: SREAConfig,
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest_path)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    inner_folds = splits["inner_folds"]
    if len(inner_folds) != 5:
        raise ValueError("the paper protocol requires exactly five inner folds")
    development = list(splits["development"])
    held_out = list(splits["held_out"])
    if set(development).intersection(held_out):
        raise ValueError("development and held-out slide IDs must be disjoint")
    torch_device = torch.device(device)
    selected_epochs, fold_results = select_epoch_count(
        records, inner_folds, model_name, cfg, protocol, torch_device, output
    )
    refit = refit_and_evaluate(
        records,
        development,
        held_out,
        selected_epochs,
        model_name,
        cfg,
        protocol,
        torch_device,
        output,
    )
    summary = {
        "model_name": model_name,
        "model_config": asdict(cfg),
        "protocol": protocol,
        "selected_refit_epochs": selected_epochs,
        "inner_folds": fold_results,
        **refit,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary
