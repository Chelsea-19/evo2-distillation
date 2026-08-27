from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "00_run_all_phase4_5.ipynb"


def test_master_notebook_is_current_and_valid_python() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_master_notebook.py"), "--check"],
        check=True,
    )
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")


def test_master_notebook_defaults_keep_final_gate_closed_and_test_locked() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in document["cells"])
    assert 'REPO_URL = "https://github.com/Chelsea-19/evo2-distillation.git"' in source
    assert 'DRIVE_ROOT = "/content/drive/MyDrive/evo2-distillation"' in source
    assert 'FASTA_ARCHIVE_NAME = "frozen_fasta_v1.tar.gz"' in source
    assert "AUTO_RESTORE_FASTA_ARCHIVE = True" in source
    assert "restore_fasta_archive.py" in source
    assert "APPROVE_FINAL_SELECTION = False" in source
    assert "RUN_FINAL_ENSEMBLE = False" in source
    assert "PHASE 5 STARTED: NO" in source
    assert "--split\", \"test" not in source


def test_final_ensemble_can_preserve_selected_student_variant() -> None:
    trainer = (REPO_ROOT / "src" / "evo2_distill" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert 'config.get("student_variant", config["experiment_type"])' in trainer
    assert 'student_variant == "baseline_correction"' in trainer
    assert 'student_variant == "tail_aware"' in trainer
