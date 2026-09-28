from pathlib import Path

import numpy as np

from srea_mil.data import FeatureManifestDataset


def test_deterministic_instance_cap(tmp_path: Path):
    features = np.arange(40 * 1280, dtype=np.float32).reshape(40, 1280)
    np.save(tmp_path / "abnormal.npy", features)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "slide_id,label,abnormal_feature_path,normal_feature_path\n"
        "s1,1,abnormal.npy,\n",
        encoding="utf-8",
    )
    dataset = FeatureManifestDataset(manifest, max_instances=8, seed=42, epoch=3)
    first = dataset[0]["abnormal"]
    second = dataset[0]["abnormal"]
    assert first.shape == (8, 1280)
    assert np.array_equal(first.numpy(), second.numpy())
