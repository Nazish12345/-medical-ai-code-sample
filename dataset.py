"""Dataset loading with reproducible ordering.

Two sources are supported:
  * ``synthetic`` - a seeded random toy dataset; needs no download.
  * ``cifar10``   - the public CIFAR-10 test split via torchvision.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def make_synthetic(
    num_samples: int = 512,
    num_classes: int = 10,
    image_size: int = 32,
    seed: int = 0,
) -> Dataset:
    """Random images whose class shifts the mean intensity, so it is learnable."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_samples,), generator=gen)
    images = torch.randn(num_samples, 3, image_size, image_size, generator=gen)
    images += (labels.float() / num_classes).view(-1, 1, 1, 1)
    return TensorDataset(images, labels)


def make_cifar10(root: str = "data", download: bool = False) -> Dataset:
    from torchvision import datasets, transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return datasets.CIFAR10(root=root, train=False, transform=tfm, download=download)


def build_dataset(name: str, **kwargs) -> Dataset:
    key = name.strip().lower()
    if key == "synthetic":
        return make_synthetic(**kwargs)
    if key == "cifar10":
        return make_cifar10(**kwargs)
    raise KeyError(f"unknown dataset {name!r}; have ['cifar10', 'synthetic']")


def build_loader(dataset: Dataset, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
    """Evaluation loader: fixed order (no shuffle), so results are reproducible."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
