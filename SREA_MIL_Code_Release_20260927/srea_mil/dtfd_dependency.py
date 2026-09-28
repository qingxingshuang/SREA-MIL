"""Load a pinned official DTFD-MIL installation without distributing its source."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import sys


DTFD_COMMIT = "10964f4dcc27c65ce110a0e9a3b9240bff58da8a"
DTFD_FILES = {
    "network.py": "8a45fad3cdd461fc5826b21d45481001835fd9586182ee139f9000193006c30a",
    "Attention.py": "bcdc596fb8a1eae79575aa373032f1939eaf7998e885c280052d34fb396c1d34",
}
DTFD_LICENSE_SHA256 = "b46526141d2c0c93a325a5c03525fe615ac6ab3f211422bb4a99cdcb4ea478f3"
DEFAULT_DTFD_ROOT = Path.home() / ".cache" / "srea-mil" / "DTFD-MIL" / DTFD_COMMIT


def load_dtfd_layers():
    """Return the upstream layer classes after validating their pinned sources."""
    root = Path(os.environ.get("SREA_DTFD_ROOT", DEFAULT_DTFD_ROOT)).expanduser().resolve()
    expected_files = {f"Model/{name}": digest for name, digest in DTFD_FILES.items()}
    expected_files["LICENSE"] = DTFD_LICENSE_SHA256
    for relative_path, expected_hash in expected_files.items():
        source = root / relative_path
        if not source.is_file():
            raise RuntimeError(
                f"Missing official DTFD-MIL dependency at {source}. "
                "Run 'python scripts/fetch_dtfd.py' first, or set SREA_DTFD_ROOT "
                "to a checkout of the pinned upstream commit."
            )
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"DTFD-MIL source mismatch at {source}: expected SHA-256 "
                f"{expected_hash}, got {actual_hash}. Use pinned upstream "
                f"commit {DTFD_COMMIT}; local modifications are not accepted."
            )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        network = importlib.import_module("Model.network")
        attention = importlib.import_module("Model.Attention")
    except ImportError as exc:
        raise RuntimeError(f"Cannot import official DTFD-MIL from {root}") from exc
    for module, filename in ((network, "network.py"), (attention, "Attention.py")):
        if Path(module.__file__).resolve() != root / "Model" / filename:
            raise RuntimeError(
                f"A different 'Model' package was loaded from {module.__file__}; "
                "use a clean Python environment for the official DTFD-MIL dependency."
            )
    return (
        network.DimReduction,
        network.Classifier_1fc,
        attention.Attention_Gated,
        attention.Attention_with_Classifier,
    )
