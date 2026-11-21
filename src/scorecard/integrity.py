"""File fingerprints (D-020, D-023).

Kept in its own tiny module so the scoring API can verify artifacts without
importing the evaluation pipeline (which pulls in LightGBM and plotting).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path | str) -> str:
    """SHA-256 of a file, read in 1 MB chunks so large files never sit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
