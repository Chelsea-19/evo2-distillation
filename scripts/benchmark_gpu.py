#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evo2_distill.models.student import ArchitectureV1, ScalarStudentV1
from evo2_distill.training.runtime import choose_precision, environment_summary, select_device
from evo2_distill.utils.io import atomic_json_dump


def probe(model: torch.nn.Module, device: torch.device, batch_size: int, precision: str, steps: int) -> dict[str, float]:
    tokens = torch.randint(0, 5, (batch_size, 512), dtype=torch.long, device=device)
    target = torch.rand(batch_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    enabled = device.type == "cuda" and precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            loss = torch.nn.functional.huber_loss(model(tokens), target)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            loss = torch.nn.functional.huber_loss(model(tokens), target)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "windows_per_second": batch_size * steps / elapsed,
        "batches_per_second": steps / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--training-rows", type=int, default=1_640_110)
    args = parser.parse_args()
    device = select_device(args.device)
    precision = choose_precision(device, args.precision)
    model = ScalarStudentV1(ArchitectureV1()).to(device)
    candidates = [64, 128, 256, 512, 1024, 2048] if device.type == "cuda" else [16, 32, 64, 128]
    successful: list[tuple[int, dict[str, float]]] = []
    for batch_size in candidates:
        try:
            result = probe(model, device, batch_size, precision, args.steps)
            successful.append((batch_size, result))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    if not successful:
        raise SystemExit("No benchmark batch size succeeded")
    # Conservative backoff: keep the fastest successful candidate below the first OOM.
    chosen_batch, metrics = max(successful, key=lambda item: item[1]["windows_per_second"])
    env = environment_summary()
    payload = {
        "benchmark_only_not_scientific_performance": True,
        "gpu_name": env["gpu"],
        "vram_gb": env["vram_gb"],
        "precision": precision,
        "batch_size": chosen_batch,
        "num_workers": args.num_workers,
        "windows_per_second": metrics["windows_per_second"],
        "batches_per_second": metrics["batches_per_second"],
        "estimated_epoch_seconds": args.training_rows / metrics["windows_per_second"],
        "peak_vram_gb": metrics["peak_vram_gb"],
        "token_cache_mode": "local_memmap_expected",
        "test_accessed": False,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

