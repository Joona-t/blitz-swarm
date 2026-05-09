"""Shared pytest fixtures for blitz-swarm tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure blitz-swarm package root is importable
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def slate_path(repo_root: Path) -> Path:
    return repo_root / "bench" / "slate_v1.toml"
