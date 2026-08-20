"""Shared test helpers.

External-data policy (TESTING_STRATEGY.md): tests needing manually provided
datasets call :func:`external_data_path` and skip with a clear message when
the data is absent. Nothing is ever downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

EXTERNAL_DATA_ENV = "EMBODIED_SYNC_EXTERNAL_DATA_ROOT"


def external_data_path(subdir: str) -> Path:
    """Return the local path for an external dataset or skip the test.

    ``subdir`` is one of: umi, lerobot, mcap, xdf, surg_sync, rerun.
    """
    root = os.environ.get(EXTERNAL_DATA_ENV)
    if not root:
        pytest.skip(
            f"external dataset skipped: {EXTERNAL_DATA_ENV} is not set. "
            f"Point it at your data/external/ directory to enable this test "
            f"(see TESTING_STRATEGY.md)."
        )
    path = Path(root) / subdir
    if not path.exists():
        pytest.skip(
            f"external dataset skipped: {path} does not exist. "
            f"Manually place the dataset under {EXTERNAL_DATA_ENV}/{subdir}/ "
            f"(see TESTING_STRATEGY.md). Nothing is downloaded automatically."
        )
    return path
