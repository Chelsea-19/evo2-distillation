from pathlib import Path

import torch

from evo2_distill.models.student import ArchitectureV1, ScalarStudentV1
from evo2_distill.training.checkpoint import capture_random_states, load_checkpoint, save_checkpoint_atomic


def test_dummy_checkpoint_resume(tmp_path: Path) -> None:
    model = ScalarStudentV1(ArchitectureV1())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "epoch": 0, "global_step": 1,
        "seed": 11, "config_hash": "dummy", "best_validation_metric": 0.1,
        "random_states": capture_random_states(),
    }, path)
    restored = load_checkpoint(path, model, optimizer, scheduler)
    assert restored["global_step"] == 1

