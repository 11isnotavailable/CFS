"""Fine-tune one fixed-width ViT under a chosen attention-head partition.

This is a head-granularity control, not parameter scaling: QKV/WO shapes and
the leading projection FLOPs remain fixed as the number of heads changes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config
from torchvision.datasets import ImageFolder

from src.model_loading import create_timm_model, repartition_timm_attention_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(targets: list[int], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    train, holdout = [], []
    for label in np.unique(targets_array):
        indices = np.flatnonzero(targets_array == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        holdout.extend(indices[:count].tolist()); train.extend(indices[count:].tolist())
    rng.shuffle(train); rng.shuffle(holdout)
    return train, holdout


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval(); loss_sum = 0.0; correct = 0; count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images); loss = torch.nn.functional.cross_entropy(logits, targets)
        loss_sum += float(loss) * len(targets); correct += int((logits.argmax(-1) == targets).sum()); count += len(targets)
    return loss_sum / count, correct / count


def main() -> None:
    args = parse_args(); seed_everything(args.seed)
    device = torch.device(args.device)
    model, _ = create_timm_model(args.model, args.checkpoint)
    original_heads = int(model.blocks[0].attn.num_heads)
    model.reset_classifier(10)
    repartition_timm_attention_heads(model, args.num_heads)
    data_config = resolve_model_data_config(model)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    indexing_dataset = ImageFolder(args.data_root)
    train_indices, holdout_indices = stratified_indices(
        indexing_dataset.targets, args.holdout_fraction, args.seed
    )
    train_dataset = torch.utils.data.Subset(ImageFolder(args.data_root, transform=train_transform), train_indices)
    holdout_dataset = torch.utils.data.Subset(ImageFolder(args.data_root, transform=eval_transform), holdout_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader)); warmup = max(1, int(0.05 * total_steps))
    def schedule(step: int) -> float:
        if step < warmup: return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_accuracy = -1.0; history = []; global_step = 0; started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); running_loss = 0.0; seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images); loss = torch.nn.functional.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            # AMP may skip the first optimizer update after an overflow. Do not
            # advance the learning-rate schedule on a skipped update.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            running_loss += float(loss.detach()) * len(targets); seen += len(targets); global_step += 1
        validation_loss, validation_accuracy = evaluate(model, holdout_loader, device)
        row = {
            "epoch": epoch + 1, "train_loss": running_loss / seen,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
            "elapsed_minutes": (time.perf_counter() - started) / 60,
        }
        history.append(row); print(json.dumps(row), flush=True)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({
                "model": args.model, "num_classes": 10,
                "class_to_idx": indexing_dataset.class_to_idx,
                "state_dict": model.state_dict(), "trained_num_heads": args.num_heads,
                "original_checkpoint_heads": original_heads, "seed": args.seed,
                "training_protocol": vars(args) | {"data_root": str(args.data_root), "checkpoint": str(args.checkpoint), "output": str(args.output)},
                "history": history,
            }, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "best_validation_accuracy": best_accuracy, "history": history,
        "train_examples": len(train_dataset), "holdout_examples": len(holdout_dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "scientific_scope": "fixed-width head-partition control, not parameter scaling",
    }, indent=2), encoding="utf-8")
    print(f"saved={args.output} best_accuracy={best_accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
