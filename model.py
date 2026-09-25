"""Image-classification models.

Models are created by name through a small registry so the evaluation code does
not depend on any specific architecture.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class SmallCNN(nn.Module):
    """A compact convolutional classifier for 32x32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def _resnet18(num_classes: int = 10) -> nn.Module:
    from torchvision.models import resnet18

    return resnet18(weights=None, num_classes=num_classes)


_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "small_cnn": SmallCNN,
    "resnet18": _resnet18,
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, num_classes: int = 10, checkpoint: str | None = None) -> nn.Module:
    """Build a model by name and optionally load a local state_dict."""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {available_models()}")
    model = _REGISTRY[key](num_classes=num_classes)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model
