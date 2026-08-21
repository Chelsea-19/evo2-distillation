from pathlib import Path

import pytest

from evo2_distill.data.safety import require_phase4_5_split, verify_test_lock


ROOT = Path(__file__).resolve().parents[1]


def test_test_access_rejected() -> None:
    assert require_phase4_5_split("development") == "development"
    assert require_phase4_5_split("validation") == "validation"
    with pytest.raises(PermissionError):
        require_phase4_5_split("test")


def test_frozen_lock_is_present() -> None:
    verify_test_lock(ROOT / "manifests" / "TEST_SET_LOCKED.txt")


def test_no_unlock_implementation() -> None:
    source = (ROOT / "src" / "evo2_distill" / "data" / "safety.py").read_text(encoding="utf-8")
    assert "PHASE4_5_ALLOWED_SPLITS" in source
    assert "test\"}" not in source

