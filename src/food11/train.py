"""Train a ResNet-18 classifier for Food-11 and track it with MLflow."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DATASETS = {
    "processed": Path("data/food11_processed"),
    "mini": Path("data/food11_processed_mini"),
}
SPLITS = {"train": "training", "validation": "validation", "test": "evaluation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="food11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Fine-tune all ResNet layers instead of only the classifier head.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(root: Path, batch_size: int) -> tuple[dict[str, DataLoader], list[str]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    image_sets = {
        name: datasets.ImageFolder(root / directory, transform=transform)
        for name, directory in SPLITS.items()
    }
    classes = image_sets["train"].classes
    for name, dataset in image_sets.items():
        if dataset.classes != classes:
            raise ValueError(f"Class mapping differs in {name} split")

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in image_sets.items()
    }
    return loaders, classes


def make_model(num_classes: int, train_backbone: bool) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if not train_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1:
        raise ValueError("epochs, lr, and batch-size must be positive")

    seed_everything(args.seed)
    data_root = DATASETS[args.dataset]
    loaders, classes = make_loaders(data_root, args.batch_size)
    if len(classes) != 11:
        raise ValueError(f"Expected 11 classes, found {len(classes)}: {classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(classes), args.train_backbone).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "architecture": "resnet18",
                "pretrained": True,
                "train_backbone": args.train_backbone,
                "device": str(device),
                "seed": args.seed,
            }
        )

        for epoch in range(args.epochs):
            train_loss, _ = run_epoch(
                model, loaders["train"], loss_fn, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(
                model, loaders["validation"], loss_fn, device
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy:.4f}",
                flush=True,
            )

        test_loss, test_accuracy = run_epoch(
            model, loaders["test"], loss_fn, device
        )
        mlflow.log_metrics(
            {"test_loss": test_loss, "test_accuracy": test_accuracy},
            step=args.epochs - 1,
        )
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        print(
            f"run_id={run.info.run_id} test_accuracy={test_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
