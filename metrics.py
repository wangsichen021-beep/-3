from __future__ import annotations

import torch
from torch.nn import functional as F


class SegmentationMetrics:
    """Streaming metrics for multi-class semantic segmentation."""

    def __init__(self, num_classes: int, ap_bins: int = 101) -> None:
        self.num_classes = num_classes
        self.ap_bins = ap_bins
        self.confusion = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.float64,
        )
        self.ap_counts = torch.zeros((num_classes, ap_bins), dtype=torch.float64)
        self.ap_positives = torch.zeros((num_classes, ap_bins), dtype=torch.float64)

    def reset(self) -> None:
        self.confusion.zero_()
        self.ap_counts.zero_()
        self.ap_positives.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = F.softmax(logits.float(), dim=1).detach().cpu()
        pred = torch.argmax(probs, dim=1).view(-1)
        target = target.detach().cpu().view(-1)
        valid = (target >= 0) & (target < self.num_classes)
        target_valid = target[valid]
        pred_valid = pred[valid]

        encoded = self.num_classes * target_valid + pred_valid
        hist = torch.bincount(
            encoded,
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist.to(dtype=torch.float64)

        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)[valid]
        for class_idx in range(self.num_classes):
            scores = probs_flat[:, class_idx]
            labels = (target_valid == class_idx).to(dtype=torch.float64)
            bins = torch.clamp(
                (scores * (self.ap_bins - 1)).long(),
                min=0,
                max=self.ap_bins - 1,
            )
            self.ap_counts[class_idx] += torch.bincount(
                bins,
                minlength=self.ap_bins,
            ).to(dtype=torch.float64)
            self.ap_positives[class_idx] += torch.bincount(
                bins,
                weights=labels,
                minlength=self.ap_bins,
            ).to(dtype=torch.float64)

    def compute(self) -> dict[str, object]:
        true_positive = torch.diag(self.confusion)
        predicted = self.confusion.sum(dim=0)
        actual = self.confusion.sum(dim=1)
        union = predicted + actual - true_positive
        class_iou = true_positive / torch.clamp(union, min=1.0)
        present = union > 0
        miou = class_iou[present].mean().item() if present.any() else 0.0

        total_pixels = torch.clamp(self.confusion.sum(), min=1.0)
        pixel_accuracy = (true_positive.sum() / total_pixels).item()

        class_ap = []
        for class_idx in range(self.num_classes):
            counts = torch.flip(self.ap_counts[class_idx], dims=(0,))
            positives = torch.flip(self.ap_positives[class_idx], dims=(0,))
            total_positives = positives.sum()
            if total_positives <= 0:
                class_ap.append(0.0)
                continue
            tp_cumsum = torch.cumsum(positives, dim=0)
            fp_cumsum = torch.cumsum(counts - positives, dim=0)
            precision = tp_cumsum / torch.clamp(tp_cumsum + fp_cumsum, min=1.0)
            recall_delta = positives / total_positives
            class_ap.append(torch.sum(precision * recall_delta).item())
        mean_ap = float(sum(class_ap) / max(len(class_ap), 1))

        return {
            "miou": miou,
            "class_iou": class_iou.tolist(),
            "pixel_accuracy": pixel_accuracy,
            "map": mean_ap,
            "class_ap": class_ap,
        }


class MeanIoU:
    """Compatibility wrapper used by the smoke test."""

    def __init__(self, num_classes: int) -> None:
        self.metrics = SegmentationMetrics(num_classes=num_classes)

    def reset(self) -> None:
        self.metrics.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self.metrics.update(logits, target)

    def compute(self) -> tuple[float, list[float]]:
        computed = self.metrics.compute()
        return computed["miou"], computed["class_iou"]
