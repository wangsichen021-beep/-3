from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLASS_NAMES = ("pet", "border", "background")


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    image_size: int = 160
    val_ratio: float = 0.2
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    download: bool = True


class PetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet trimap dataset for 3-class semantic segmentation.

    The original trimap masks contain labels 1, 2, 3. CrossEntropyLoss expects
    labels 0, 1, 2, so masks are shifted by -1.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        train: bool,
        download: bool,
    ) -> None:
        self.dataset = OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.train = train
        self.image_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )
        self.normalize = transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        if isinstance(mask, tuple):
            mask = mask[0]

        image = image.convert("RGB")
        mask = mask.convert("L")

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.train:
            image = self.color_jitter(image)

        image = self.image_resize(image)
        mask = self.mask_resize(mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)

        mask_array = np.asarray(mask, dtype=np.int64) - 1
        mask_array = np.clip(mask_array, 0, len(CLASS_NAMES) - 1)
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor, mask_tensor


def build_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    root = Path(config.root)
    train_full = PetSegmentationDataset(
        root=root,
        split="trainval",
        image_size=config.image_size,
        train=True,
        download=config.download,
    )
    val_full = PetSegmentationDataset(
        root=root,
        split="trainval",
        image_size=config.image_size,
        train=False,
        download=False,
    )

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_size = max(1, int(len(indices) * config.val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
