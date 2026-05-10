from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from torch import nn

from task3_segmentation.data import CLASS_NAMES, DataConfig, build_dataloaders
from task3_segmentation.losses import CombinedLoss, DiceLoss
from task3_segmentation.model import UNet
from task3_segmentation.train_utils import (
    evaluate,
    resolve_device,
    set_seed,
    train_one_epoch,
    write_history,
    write_json,
)


LOSS_CHOICES = ("ce", "dice", "ce_dice")
LOSS_NAMES = {
    "ce": "Cross-Entropy Loss",
    "dice": "Dice Loss",
    "ce_dice": "Cross-Entropy Loss + Dice Loss",
}


def build_criterion(loss_name: str, num_classes: int) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "dice":
        return DiceLoss(num_classes=num_classes)
    if loss_name == "ce_dice":
        return CombinedLoss(num_classes=num_classes)
    raise ValueError(f"Unsupported loss: {loss_name}")


def run_experiment(
    args: argparse.Namespace,
    loss_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, object]:
    num_classes = len(CLASS_NAMES)
    set_seed(args.seed)

    run_dir = Path(args.runs_dir) / loss_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=args.base_channels,
    ).to(device)
    criterion = build_criterion(loss_name, num_classes=num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    wandb_run = None
    if args.wandb:
        try:
            wandb_env_dir = Path(args.runs_dir) / ".wandb_env"
            wandb_env_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("WANDB_MODE", args.wandb_mode)
            os.environ.setdefault("WANDB_DATA_DIR", str(wandb_env_dir / "data"))
            os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_env_dir / "cache"))
            os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_env_dir / "config"))
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"task3_{loss_name}",
                mode=args.wandb_mode,
                dir=str(run_dir),
                reinit=True,
                config={
                    **vars(args),
                    "loss_name": loss_name,
                    "loss_label": LOSS_NAMES[loss_name],
                    "num_classes": num_classes,
                    "class_names": CLASS_NAMES,
                },
            )
        except Exception as exc:
            print(f"wandb logging disabled for {loss_name}: {exc}")
            wandb_run = None

    best_miou = -1.0
    best_epoch = 0
    best_class_iou: list[float] = [0.0] * num_classes
    best_class_ap: list[float] = [0.0] * num_classes
    best_pixel_accuracy = 0.0
    best_map = 0.0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            amp=args.amp,
            limit_batches=args.limit_train_batches,
        )
        val_loss, metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            amp=args.amp,
            limit_batches=args.limit_val_batches,
        )
        val_miou = float(metrics["miou"])
        class_iou = list(metrics["class_iou"])
        val_pixel_accuracy = float(metrics["pixel_accuracy"])
        val_map = float(metrics["map"])
        class_ap = list(metrics["class_ap"])
        scheduler.step()

        row = {
            "epoch": epoch,
            "loss": loss_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_pixel_accuracy": val_pixel_accuracy,
            "val_map": val_map,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for class_name, iou in zip(CLASS_NAMES, class_iou):
            row[f"iou_{class_name}"] = iou
        for class_name, ap in zip(CLASS_NAMES, class_ap):
            row[f"ap_{class_name}"] = ap
        history.append(row)
        write_history(run_dir / "history.csv", history)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)

        print(
            f"[{loss_name}] epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f} val_acc={val_pixel_accuracy:.4f} "
            f"val_mAP={val_map:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            best_class_iou = class_iou
            best_class_ap = class_ap
            best_pixel_accuracy = val_pixel_accuracy
            best_map = val_map
            torch.save(
                {
                    "epoch": epoch,
                    "loss": loss_name,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_miou": val_miou,
                    "val_pixel_accuracy": val_pixel_accuracy,
                    "val_map": val_map,
                    "class_iou": class_iou,
                    "class_ap": class_ap,
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                },
                run_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "loss": loss_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "args": vars(args),
        },
        run_dir / "last.pt",
    )
    write_history(run_dir / "history.csv", history)

    result = {
        "loss": loss_name,
        "loss_label": LOSS_NAMES[loss_name],
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_val_pixel_accuracy": best_pixel_accuracy,
        "best_val_map": best_map,
    }
    for class_name, iou in zip(CLASS_NAMES, best_class_iou):
        result[f"best_iou_{class_name}"] = iou
    for class_name, ap in zip(CLASS_NAMES, best_class_ap):
        result[f"best_ap_{class_name}"] = ap

    write_json(run_dir / "result.json", result)
    if wandb_run is not None:
        wandb_run.summary["best_val_miou"] = best_miou
        wandb_run.summary["best_val_pixel_accuracy"] = best_pixel_accuracy
        wandb_run.summary["best_val_map"] = best_map
        wandb_run.finish()
    return result


def write_comparison(runs_dir: Path, results: list[dict[str, object]]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    md_lines = [
        "# Task 3 Loss Comparison",
        "",
        "| Loss | Best Epoch | Val mIoU | Pixel Acc | Val mAP | Pet IoU | Border IoU | Background IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        md_lines.append(
            "| {loss_label} | {best_epoch} | {best_val_miou:.4f} | "
            "{best_val_pixel_accuracy:.4f} | {best_val_map:.4f} | "
            "{best_iou_pet:.4f} | {best_iou_border:.4f} | "
            "{best_iou_background:.4f} |".format(**row)
        )
    (runs_dir / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch U-Net on Oxford-IIIT Pet trimaps.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--runs-dir", default="runs/task3")
    parser.add_argument("--loss", choices=("all", *LOSS_CHOICES), default="all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="task3-unet-segmentation")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="offline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"classes: {', '.join(CLASS_NAMES)}")

    data_config = DataConfig(
        root=args.data_root,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
    )
    train_loader, val_loader = build_dataloaders(data_config)
    print(
        f"train images: {len(train_loader.dataset)}, "
        f"val images: {len(val_loader.dataset)}"
    )

    losses = list(LOSS_CHOICES) if args.loss == "all" else [args.loss]
    results = []
    for loss_name in losses:
        print(f"\n=== {LOSS_NAMES[loss_name]} ===")
        results.append(
            run_experiment(
                args=args,
                loss_name=loss_name,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
        )

    write_comparison(Path(args.runs_dir), results)
    print(f"\nSaved comparison to {Path(args.runs_dir) / 'comparison.md'}")


if __name__ == "__main__":
    main()
