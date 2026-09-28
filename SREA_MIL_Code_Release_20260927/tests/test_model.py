import torch
import pytest

from srea_mil.model import MODEL_SPECS, build_model
from srea_mil.dtfd_dependency import DTFD_FILES


def test_missing_official_dependency_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("SREA_DTFD_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="fetch_dtfd.py"):
        build_model()


def test_modified_official_dependency_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("SREA_DTFD_ROOT", str(tmp_path))
    model_dir = tmp_path / "Model"
    model_dir.mkdir()
    for name in DTFD_FILES:
        (model_dir / name).write_text("# modified", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source mismatch"):
        build_model()


def test_all_models_forward_and_backward():
    torch.manual_seed(1)
    abnormal = torch.randn(2, 16, 1280)
    normal = torch.randn(2, 16, 1280)
    mask = torch.ones(2, 16, dtype=torch.bool)
    labels = torch.tensor([0, 1])
    for name in MODEL_SPECS:
        model = build_model(name)
        output = model(abnormal, mask, normal, mask, labels)
        assert output["logits"].shape == (2, 2)
        loss = torch.nn.functional.cross_entropy(output["logits"], labels)
        loss = loss + 0.5 * output["auxiliary_loss"]
        assert torch.isfinite(loss)
        loss.backward()
