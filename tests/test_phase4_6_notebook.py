from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_phase4_6_rank_tail.ipynb"


def test_phase4_6_notebook_is_generated_and_valid_python() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_phase4_6_notebook.py"), "--check"], check=True)
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"phase4_6_cell_{index}")


def test_phase4_6_defaults_are_validation_only_and_gate_is_closed() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in document["cells"])
    assert 'CANDIDATE_VARIANTS = ["rank_tail_balanced", "rank_tail_focus"]' in source
    assert "APPROVE_PHASE4_6_SELECTION = False" in source
    assert "RUN_FIVE_MEMBER_ENSEMBLE = False" in source
    assert 'assert config.get("test_accessed") is False' in source
    assert "TEST REMAINS LOCKED: YES" in source
    assert "PHASE 5 STARTED: NO" in source
