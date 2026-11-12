"""Train a deterministic Imagenette linear probe on a frozen DINO ViT-S.

The saved checkpoint contains a standard timm ViT with a 10-class head, so the
existing intervention code can use it without changing the transformer graph.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import ImageFolder

from scripts.run_experiment0 import configure_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--val-indices-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="vit_small_patch16_224.dino",
    )
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def extract_features(
    backbone: nn.Module,
    dataset: ImageFolder,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    features = []
    labels = []
    backbone.eval()
    with torch.inference_mode():
        for batch_index, (image, target) in enumerate(loader):
            image = image.to(device=device, dtype=torch.float16)
            value = backbone(image).float().cpu()
            features.append(value)
            labels.append(target)
            print(f"features={batch_index + 1}/{len(loader)}")
    return torch.cat(features), torch.cat(labels)


def accuracy(
    probe: nn.Module,
    feature: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> float:
    probe.eval()
    with torch.inference_mode():
        prediction = probe(feature.to(device)).argmax(dim=-1).cpu()
    return float((prediction == target).float().mean())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_cache()
    import timm

    device = torch.device(args.device)
    backbone = (
        timm.create_model(args.model, pretrained=True, num_classes=0)
        .eval()
        .to(device=device, dtype=torch.float16)
    )
    transform = create_transform(
        **resolve_model_data_config(backbone),
        is_training=False,
    )
    train_dataset = ImageFolder(args.train_root, transform=transform)
    val_dataset = ImageFolder(args.val_root, transform=transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Train/validation class mappings differ.")

    train_feature, train_target = extract_features(
        backbone,
        train_dataset,
        args.feature_batch_size,
        device,
    )
    val_metadata = json.loads(
        args.val_indices_metadata.read_text(encoding="utf-8")
    )
    val_indices = [int(index) for index in val_metadata["sample_indices"]]
    if any(index < 0 or index >= len(val_dataset) for index in val_indices):
        raise ValueError("Validation metadata indices are outside the dataset.")
    val_subset = Subset(val_dataset, val_indices)
    val_feature, val_target = extract_features(
        backbone,
        val_subset,
        args.feature_batch_size,
        device,
    )
    mean = train_feature.mean(dim=0)
    std = train_feature.std(dim=0).clamp_min(1e-5)
    train_feature = (train_feature - mean) / std
    val_feature = (val_feature - mean) / std

    probe = nn.Linear(
        train_feature.shape[-1],
        len(train_dataset.classes),
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(train_feature, train_target),
        batch_size=args.probe_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_accuracy = -np.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        probe.train()
        for feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(feature.to(device))
            loss = nn.functional.cross_entropy(
                logits,
                target.to(device),
            )
            loss.backward()
            optimizer.step()
        val_accuracy = accuracy(
            probe,
            val_feature,
            val_target,
            device,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"val_accuracy={100 * val_accuracy:.2f}%"
            )

    if best_state is None:
        raise RuntimeError("No linear-probe checkpoint was selected.")
    probe.load_state_dict(best_state)

    # Calibrate a single temperature on the same tune-only split used for
    # checkpoint selection. The future oracle test remains untouched.
    with torch.no_grad():
        tune_logits = probe(val_feature.to(device)).detach()
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    calibrator = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )
    tune_target_device = val_target.to(device)

    def calibration_closure() -> torch.Tensor:
        calibrator.zero_grad()
        temperature = log_temperature.exp()
        loss = nn.functional.cross_entropy(
            tune_logits / temperature,
            tune_target_device,
        )
        loss.backward()
        return loss

    uncalibrated_nll = float(
        nn.functional.cross_entropy(
            tune_logits,
            tune_target_device,
        )
    )
    calibrator.step(calibration_closure)
    temperature = float(log_temperature.detach().exp())
    calibrated_nll = float(
        nn.functional.cross_entropy(
            tune_logits / temperature,
            tune_target_device,
        )
    )

    # Fold feature normalization into the linear layer so the resulting model
    # consumes the backbone's original unnormalized CLS representation. Fold
    # temperature scaling into the same head.
    weight = probe.weight.detach().cpu()
    bias = probe.bias.detach().cpu()
    folded_weight = weight / std[None, :] / temperature
    folded_bias = (
        bias - (weight * mean[None, :] / std[None, :]).sum(dim=1)
    ) / temperature

    deploy_model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=len(train_dataset.classes),
    )
    deploy_model.head.weight.data.copy_(folded_weight)
    deploy_model.head.bias.data.copy_(folded_bias)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_classes": len(train_dataset.classes),
        "class_to_idx": train_dataset.class_to_idx,
        "validation_accuracy": float(best_accuracy),
        "validation_indices_metadata": str(
            args.val_indices_metadata.resolve()
        ),
        "temperature": temperature,
        "tune_nll_uncalibrated": uncalibrated_nll,
        "tune_nll_calibrated": calibrated_nll,
        "seed": args.seed,
        "state_dict": deploy_model.state_dict(),
    }
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "state_dict"
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"best_val_accuracy={100 * best_accuracy:.2f}%")
    print(
        f"temperature={temperature:.4f} "
        f"tune_nll={uncalibrated_nll:.4f}->{calibrated_nll:.4f}"
    )
    print(f"checkpoint={args.output.resolve()}")


if __name__ == "__main__":
    main()
