from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from evo2_distill.data.dataset import GenomePairBatchSampler, TokenWindowDataset
from evo2_distill.evaluation.ranking import evaluate_validation_predictions
from evo2_distill.losses.ranking import pairwise_logistic_ranking_loss
from evo2_distill.models.student import ArchitectureV1, ScalarStudentV1, trainable_parameter_count
from evo2_distill.training.checkpoint import capture_random_states, load_checkpoint, save_checkpoint_atomic
from evo2_distill.training.runtime import autocast_context, choose_precision, environment_summary, seed_everything, select_device
from evo2_distill.utils.io import atomic_json_dump, sha256_file


def _config_hash(config: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNCOMMITTED_PACKAGE"


def _path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _architecture(config: dict) -> ArchitectureV1:
    values = dict(config.get("architecture", {}))
    if "dilation_schedule" in values:
        values["dilation_schedule"] = tuple(values["dilation_schedule"])
    return ArchitectureV1(**values)


def _subset_development(dataset: TokenWindowDataset, fraction: float, seed: int) -> None:
    if not (0 < fraction <= 1):
        raise ValueError("data_fraction must be in (0, 1]")
    if fraction == 1:
        return
    rng = np.random.default_rng(seed)
    positions: list[int] = []
    for _, group in dataset.frame.groupby("assembly_id", sort=True):
        count = max(2, int(np.ceil(fraction * len(group))))
        positions.extend(rng.choice(group.index.to_numpy(), size=min(count, len(group)), replace=False).tolist())
    dataset.frame = dataset.frame.loc[sorted(positions)].reset_index(drop=True)
    dataset.refresh_assembly_codes()


def _apply_baseline_correction(dataset: TokenWindowDataset, model_path: Path) -> None:
    estimator = joblib.load(model_path)
    features = dataset.frame[["gc_content", "window_length", "k4_rarity", "k6_rarity"]]
    baseline = estimator.predict(features)
    dataset.frame["absolute_residual_original"] = dataset.frame[dataset.target_column]
    dataset.frame["baseline_prediction"] = baseline
    dataset.frame[dataset.target_column] = dataset.frame[dataset.target_column] - baseline


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    dataset: TokenWindowDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    label: str,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model.eval()
    predictions: list[np.ndarray] = []
    total_batches = len(loader)
    progress_every = max(1, total_batches // 4)
    processed = 0
    started = time.perf_counter()
    print(f"[{label}] prediction batches={total_batches:,} rows={len(dataset):,}", flush=True)
    for batch_index, batch in enumerate(loader, start=1):
        tokens = batch["tokens"].to(device=device, dtype=torch.long, non_blocking=True)
        values = model(tokens).float().cpu().numpy()
        predictions.append(values)
        processed += len(tokens)
        if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[{label}] {batch_index:,}/{total_batches:,} batches "
                f"({100 * batch_index / total_batches:.1f}%) "
                f"{processed / elapsed:,.0f} windows/s",
                flush=True,
            )
    frame = dataset.frame[["window_id", "assembly_id", "cluster_id", dataset.target_column]].copy()
    frame = frame.rename(columns={dataset.target_column: "teacher_target"})
    frame["prediction"] = np.concatenate(predictions)
    if "baseline_prediction" in dataset.frame:
        frame["teacher_target"] = dataset.frame["absolute_residual_original"].to_numpy()
        frame["prediction"] = frame["prediction"] + dataset.frame["baseline_prediction"].to_numpy()
    frame["split"] = "validation"
    return frame


def run_training(config_path: str | Path, resume_path: str | Path | None = None) -> Path:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("validation_only") is not True or config.get("test_accessed") is not False:
        raise ValueError("Phase 4.5 config must declare validation_only=true and test_accessed=false")
    paths = config["paths"]
    cache_dir = _path(paths["cache_dir"])
    run_root = _path(paths["run_root"])
    run_id = config.get("run_id") or f"{config['experiment_type']}_seed{config['seed']}_{int(time.time())}"
    if resume_path:
        run_dir = Path(resume_path).resolve().parents[1]
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory missing: {run_dir}")
    else:
        run_dir = run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoint").mkdir()
        (run_dir / "predictions").mkdir()
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"[setup] run_id={run_id} config={config_path}", flush=True)
    print(f"[setup] cache={cache_dir}", flush=True)

    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(config["infrastructure"].get("device", "auto"))
    precision = choose_precision(device, config["infrastructure"].get("precision", "auto"))
    batch_size = int(config["infrastructure"]["batch_size"])
    workers = int(config["infrastructure"].get("num_workers", 2))

    extra = ["gc_content", "window_length", "k4_rarity", "k6_rarity"]
    print("[setup] loading development metadata", flush=True)
    train_data = TokenWindowDataset(cache_dir, "development", additional_columns=extra, include_metadata=False)
    print("[setup] loading validation metadata", flush=True)
    validation_data = TokenWindowDataset(cache_dir, "validation", additional_columns=extra, include_metadata=False)
    _subset_development(train_data, float(config.get("data_fraction", 1.0)), seed)
    student_variant = config.get("student_variant", config["experiment_type"])
    if student_variant == "baseline_correction":
        baseline_model = _path(paths["baseline_model"])
        _apply_baseline_correction(train_data, baseline_model)
        _apply_baseline_correction(validation_data, baseline_model)
        train_data.refresh_assembly_codes()
        validation_data.refresh_assembly_codes()

    sampler = GenomePairBatchSampler(train_data.assembly_codes, batch_size, seed)
    loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    architecture = _architecture(config)
    model = ScalarStudentV1(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    environment = environment_summary()
    print(
        f"[setup] device={device} gpu={environment['gpu']} precision={precision} "
        f"batch_size={batch_size} workers={workers}",
        flush=True,
    )
    print(
        f"[setup] train_rows={len(train_data):,} validation_rows={len(validation_data):,} "
        f"steps_per_epoch={len(loader):,} parameters={trainable_parameter_count(model):,}",
        flush=True,
    )
    start_epoch, global_step, best_metric = 0, 0, -float("inf")
    if resume_path:
        restored = load_checkpoint(resume_path, model, optimizer, scheduler)
        if restored.get("config_hash") != _config_hash(config):
            raise ValueError("Checkpoint config hash differs from requested resume config")
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best_metric = float(restored["best_validation_metric"])

    history: list[dict[str, float | int]] = []
    ranking_weight = float(config["training"]["ranking_weight"])
    huber_weight = float(config["training"].get("huber_weight", 1.0))
    epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.perf_counter()
        running_loss = torch.zeros((), device=device)
        batch_count = 0
        total_batches = len(loader)
        progress_every = max(1, total_batches // 10)
        print(f"[train] epoch {epoch + 1}/{epochs} started", flush=True)
        for batch_index, batch in enumerate(loader, start=1):
            tokens = batch["tokens"].to(device=device, dtype=torch.long, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                predictions = model(tokens)
                per_item_huber = torch.nn.functional.huber_loss(predictions, targets, reduction="none")
                if student_variant == "tail_aware":
                    threshold = torch.quantile(targets.detach(), float(config["training"].get("tail_quantile", 0.9)))
                    weights = torch.where(targets >= threshold, float(config["training"].get("tail_weight", 2.0)), 1.0)
                    huber = (per_item_huber * weights).mean()
                else:
                    huber = per_item_huber.mean()
                rank = pairwise_logistic_ranking_loss(predictions[0::2], predictions[1::2], targets[0::2], targets[1::2])
                loss = huber_weight * huber + ranking_weight * rank
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            running_loss += loss.detach()
            batch_count += 1
            if batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches:
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                rate = batch_index * batch_size / elapsed
                remaining = (total_batches - batch_index) * batch_size / max(rate, 1e-9)
                mean_loss = float((running_loss / batch_count).item())
                print(
                    f"[train] epoch {epoch + 1}/{epochs} {batch_index:,}/{total_batches:,} "
                    f"({100 * batch_index / total_batches:.1f}%) loss={mean_loss:.6f} "
                    f"rate={rate:,.0f} windows/s eta={remaining / 60:.1f} min",
                    flush=True,
                )
        scheduler.step()

        validation_predictions = _predict(
            model, validation_data, device, batch_size, workers, f"validation epoch {epoch + 1}/{epochs}"
        )
        print(f"[validation epoch {epoch + 1}/{epochs}] calculating ranking metrics", flush=True)
        _, summary = evaluate_validation_predictions(validation_predictions)
        metric = float(summary["mean_within_genome_spearman"])
        best_metric = max(best_metric, metric)
        epoch_seconds = time.perf_counter() - epoch_start
        training_loss = float((running_loss / max(batch_count, 1)).item())
        history.append({"epoch": epoch, "global_step": global_step, "training_loss": training_loss, "validation_spearman": metric, "epoch_seconds": epoch_seconds})
        print(
            f"[epoch complete] {epoch + 1}/{epochs} loss={training_loss:.6f} "
            f"validation_spearman={metric:.6f} seconds={epoch_seconds:.1f}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "seed": seed,
            "config_hash": _config_hash(config),
            "best_validation_metric": best_metric,
            "random_states": capture_random_states(),
        }
        save_checkpoint_atomic(checkpoint, run_dir / "checkpoint" / f"epoch_{epoch + 1:03d}.pt")

    validation_predictions = _predict(model, validation_data, device, batch_size, workers, "final validation")
    validation_predictions.to_parquet(run_dir / "predictions" / "validation_predictions.parquet", index=False)
    per_genome, summary = evaluate_validation_predictions(validation_predictions)
    per_genome.to_csv(run_dir / "metrics.csv", index=False)
    atomic_json_dump(summary, run_dir / "metrics.json")
    env = environment_summary()
    manifest = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[4]),
        "config_sha256": sha256_file(run_dir / "config.yaml"),
        "dataset_manifest_sha256": sha256_file(_path(paths["dataset_manifest"])),
        "split_manifest_sha256": sha256_file(_path(paths["split_manifest"])),
        "target_sha256": sha256_file(_path(paths["target_path"])),
        "token_cache_manifest_sha256": sha256_file(cache_dir / "token_cache_manifest.json"),
        "seed": seed,
        "device": str(device),
        "GPU": env["gpu"],
        "VRAM": env["vram_gb"],
        "PyTorch": env["pytorch"],
        "CUDA": env["cuda_runtime"],
        "precision": precision,
        "batch_size": batch_size,
        "num_workers": workers,
        "training_rows": len(train_data),
        "validation_rows": len(validation_data),
        "parameter_count": trainable_parameter_count(model),
        "test_accessed": False,
    }
    atomic_json_dump(manifest, run_dir / "run_manifest.json")
    return run_dir
