from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from task3_segmentation.data import CLASS_NAMES, DataConfig, build_dataloaders
from task3_segmentation.losses import CombinedLoss
from task3_segmentation.model import UNet
from task3_segmentation.train_utils import evaluate, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_config = DataConfig(
        root=args.data_root,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
    )
    _, val_loader = build_dataloaders(data_config)

    model = UNet(
        in_channels=3,
        num_classes=len(CLASS_NAMES),
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion: nn.Module = CombinedLoss(num_classes=len(CLASS_NAMES))
    val_loss, metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=len(CLASS_NAMES),
        amp=args.amp,
        limit_batches=None,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"val_loss: {val_loss:.4f}")
    print(f"val_mIoU: {metrics['miou']:.4f}")
    print(f"val_pixel_accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"val_mAP: {metrics['map']:.4f}")
    for class_name, iou, ap in zip(
        CLASS_NAMES,
        metrics["class_iou"],
        metrics["class_ap"],
    ):
        print(f"{class_name}: IoU={iou:.4f}, AP={ap:.4f}")


if __name__ == "__main__":
    main()
