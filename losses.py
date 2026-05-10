from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Manual multi-class Dice loss for dense segmentation logits."""

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [N, C, H, W]")
        if target.ndim != 3:
            raise ValueError("target must have shape [N, H, W]")
        if logits.shape[1] != self.num_classes:
            raise ValueError("logits channel count does not match num_classes")

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)

        reduce_dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=reduce_dims)
        denominator = torch.sum(probs, dim=reduce_dims) + torch.sum(
            one_hot, dim=reduce_dims
        )
        dice_score = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """Cross entropy plus Dice loss."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, target) + self.dice(logits, target)
