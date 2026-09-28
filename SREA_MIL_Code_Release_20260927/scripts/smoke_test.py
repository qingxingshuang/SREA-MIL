"""Run a forward/backward smoke test for every released configuration."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srea_mil.model import MODEL_SPECS, build_model


def main() -> None:
    torch.manual_seed(42)
    batch_size, instances, channels = 2, 24, 1280
    abnormal = torch.randn(batch_size, instances, channels)
    normal = torch.randn(batch_size, instances, channels)
    mask = torch.ones(batch_size, instances, dtype=torch.bool)
    labels = torch.tensor([0, 1])
    for model_name in MODEL_SPECS:
        model = build_model(model_name)
        output = model(abnormal, mask, normal, mask, labels)
        loss = F.cross_entropy(output["logits"], labels) + 0.5 * output["auxiliary_loss"]
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss for {model_name}")
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if gradient_sum <= 0:
            raise RuntimeError(f"zero gradient for {model_name}")
        print(
            f"{model_name}: loss={float(loss.detach()):.6f}, "
            f"gradient_sum={gradient_sum:.3f}"
        )


if __name__ == "__main__":
    main()
