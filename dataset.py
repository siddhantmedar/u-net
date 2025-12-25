#!/usr/bin/env python3
"""
OxfordIIITPet dataset loading and DataLoader utilities for U-Net segmentation.
"""

from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)


class OxfordPetSegmentation(Dataset):
    """Wrapper for OxfordIIITPet segmentation with transforms."""

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def get_transforms(is_train=True):
    """Image transforms for U-Net."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((572, 572)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((572, 572)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def get_target_transforms():
    """Mask transforms for U-Net."""
    return transforms.Compose(
        [
            transforms.Resize((388, 388)),
            transforms.PILToTensor(),
        ]
    )


def get_dataloaders(
    split="all", batch_size=32, num_workers=4, test_size=0.1, val_size=0.1
):
    """
    Create DataLoaders for OxfordIIITPet segmentation with stratified sampling.

    Args:
        split: Which loader(s) to return - 'train', 'val', 'test', or 'all'
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        test_size: Fraction of data for test set (default 10%)
        val_size: Fraction of data for validation set (default 10%)

    Returns:
        Single DataLoader if split specified, or (train_loader, val_loader, test_loader) if 'all'
    """
    if split.lower() not in {"train", "val", "test", "all"}:
        raise ValueError("split not valid. should be train/val/test/all")

    print("Loading OxfordIIITPet dataset...")

    # Load both splits and combine
    trainval = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="segmentation",
        download=True,
    )
    test_split = OxfordIIITPet(
        root="./data",
        split="test",
        target_types="segmentation",
        download=True,
    )

    # Get labels for stratification (breed class: 0-36)
    # Labels are stored in _labels attribute
    trainval_labels = trainval._labels
    test_labels = test_split._labels

    # Combine indices and labels
    trainval_indices = list(range(len(trainval)))
    test_indices = list(range(len(test_split)))

    # Stratified split: trainval -> train + val
    train_idx, val_idx = train_test_split(
        trainval_indices,
        test_size=val_size / (1 - test_size),  # Adjust for proportion
        stratify=trainval_labels,
        random_state=42,
    )

    # Keep only a small portion of test set (stratified)
    _, small_test_idx = train_test_split(
        test_indices,
        test_size=test_size * len(trainval) / len(test_split),  # ~10% of total
        stratify=test_labels,
        random_state=42,
    )

    print(
        f"Stratified split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(small_test_idx)}"
    )

    # Create subsets
    train_subset = Subset(trainval, train_idx)
    val_subset = Subset(trainval, val_idx)
    test_subset = Subset(test_split, small_test_idx)

    # Wrap with transforms
    train_dataset = OxfordPetSegmentation(
        train_subset,
        transform=get_transforms(is_train=True),
        target_transform=get_target_transforms(),
    )
    val_dataset = OxfordPetSegmentation(
        val_subset,
        transform=get_transforms(is_train=False),
        target_transform=get_target_transforms(),
    )
    test_dataset = OxfordPetSegmentation(
        test_subset,
        transform=get_transforms(is_train=False),
        target_transform=get_target_transforms(),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    if split.lower() == "train":
        return train_loader
    elif split.lower() == "val":
        return val_loader
    elif split.lower() == "test":
        return test_loader
    else:
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, num_workers=0
    )

    # Test a batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes - Images: {images.shape}, Masks: {masks.shape}")
    print(masks.unique())
