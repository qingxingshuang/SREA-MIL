"""Fetch official DTFD-MIL model files and license from a pinned commit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srea_mil.dtfd_dependency import (
    DEFAULT_DTFD_ROOT,
    DTFD_COMMIT,
    DTFD_FILES,
    DTFD_LICENSE_SHA256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DTFD_ROOT)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    expected_files = {f"Model/{name}": digest for name, digest in DTFD_FILES.items()}
    expected_files["LICENSE"] = DTFD_LICENSE_SHA256
    # Check all existing files before any download so a locally edited copy is preserved.
    for relative_path, expected_hash in expected_files.items():
        target = destination / relative_path
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Refusing to overwrite a modified file: {target}")
    for relative_path, expected_hash in expected_files.items():
        target = destination / relative_path
        if target.exists():
            print(f"Verified {target}")
            continue
        url = (
            "https://raw.githubusercontent.com/hrzhang1123/DTFD-MIL/"
            f"{DTFD_COMMIT}/{relative_path}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            contents = response.read()
        digest = hashlib.sha256(contents).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"Upstream checksum mismatch for {url}: got {digest}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        print(f"Fetched and verified {target}")


if __name__ == "__main__":
    main()
