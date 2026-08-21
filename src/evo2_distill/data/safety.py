from __future__ import annotations

from pathlib import Path

PHASE4_5_ALLOWED_SPLITS = frozenset({"development", "validation"})


def require_phase4_5_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in PHASE4_5_ALLOWED_SPLITS:
        raise PermissionError(
            "Phase 4.5 permits development and validation only; TEST remains locked until an explicit future Phase 5 workflow."
        )
    return normalized


def verify_test_lock(lock_path: str | Path) -> None:
    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required TEST lock is missing: {path}")
    text = path.read_text(encoding="utf-8")
    required = ("DISSERTATION TEST SET LOCKED", "before Phase 5")
    if not all(marker in text for marker in required):
        raise RuntimeError(f"TEST lock content is invalid: {path}")

