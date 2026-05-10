from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .metrics import SegmentationMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp: bool,
    limit_batches: int | None,
) -> float:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    total_loss = 0.0
    total_items = 0
    progress = tqdm(
        loader,
        desc=f"train {epoch}",
        leave=False,
        disable=not sys.stderr.isatty(),
    )

    for batch_idx, (images, masks) in enumerate(progress, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=total_loss / max(total_items, 1))

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp: bool,
    limit_batches: int | None,
) -> tuple[float, dict[str, object]]:
    model.eval()
    metric = SegmentationMetrics(num_classes=num_classes)
    total_loss = 0.0
    total_items = 0
    progress = tqdm(
        loader,
        desc="valid",
        leave=False,
        disable=not sys.stderr.isatty(),
    )

    for batch_idx, (images, masks) in enumerate(progress, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, masks)

        metric.update(logits, masks)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1), metric.compute()


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
