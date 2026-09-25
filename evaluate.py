"""Deterministic evaluation of an image classifier.

Example:
    python evaluate.py --dataset synthetic --model small_cnn --seed 0
    python evaluate.py --dataset cifar10 --download --model resnet18 --checkpoint weights.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import build_dataset, build_loader
from model import available_models, build_model


def set_seed(seed: int) -> None:
    """Seed every RNG and force deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predictions, labels) over the whole loader."""
    model.eval()
    preds, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(preds), torch.cat(labels)


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def per_class_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> list[float]:
    out = []
    for c in range(num_classes):
        mask = labels == c
        out.append((preds[mask] == c).float().mean().item() if mask.any() else float("nan"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "cifar10"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    p.add_argument("--model", default="small_cnn", choices=available_models())
    p.add_argument("--checkpoint", default=None, help="optional local state_dict (.pt)")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.dataset == "synthetic":
        dataset = build_dataset("synthetic", num_classes=args.num_classes, seed=args.seed)
    else:
        dataset = build_dataset("cifar10", root=args.data_root, download=args.download)
    loader = build_loader(dataset, batch_size=args.batch_size)

    model = build_model(args.model, num_classes=args.num_classes, checkpoint=args.checkpoint).to(device)
    preds, labels = predict(model, loader, device)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "num_samples": len(labels),
        "accuracy": round(accuracy(preds, labels), 4),
        "per_class_accuracy": [round(a, 4) for a in per_class_accuracy(preds, labels, args.num_classes)],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
