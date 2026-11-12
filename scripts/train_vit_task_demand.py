"""Fine-tune one ViT on paired coarse/fine labels over identical images."""

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

from src.model_loading import create_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--task-metadata", type=Path, required=True)
    parser.add_argument("--task", choices=("coarse", "fine"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--effective-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, help="Smoke-test cap per epoch")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(
    fine_targets: list[int], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split by original fine labels, identically for coarse and fine tasks."""
    rng = np.random.default_rng(seed)
    values = np.asarray(fine_targets)
    train, holdout = [], []
    for label in np.unique(values):
        indices = np.flatnonzero(values == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        holdout.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ImageFolder, wnid_to_target: dict[str, int]):
        self.dataset = dataset
        self.target_by_fine_index = {
            fine_index: int(wnid_to_target[wnid])
            for wnid, fine_index in dataset.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, fine_target = self.dataset[index]
        return image, self.target_by_fine_index[int(fine_target)]


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        loss_sum += float(loss) * len(targets)
        correct += int((logits.argmax(-1) == targets).sum())
        count += len(targets)
    return loss_sum / count, correct / count


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.effective_batch_size % args.batch_size:
        raise ValueError("effective-batch-size must be divisible by batch-size")
    accumulation = args.effective_batch_size // args.batch_size
    device = torch.device(args.device)
    metadata = json.loads(args.task_metadata.read_text(encoding="utf-8"))
    mapping = {
        str(key): int(value)
        for key, value in metadata[f"{args.task}_class_to_idx"].items()
    }
    num_classes = len(set(mapping.values()))

    model, _ = create_timm_model(args.model, args.checkpoint)
    original_heads = int(model.blocks[0].attn.num_heads)
    model.reset_classifier(num_classes)
    config = resolve_model_data_config(model)
    train_transform = create_transform(**config, is_training=True)
    eval_transform = create_transform(**config, is_training=False)

    indexing = ImageFolder(args.train_root)
    if set(indexing.classes) != set(mapping):
        raise ValueError("Training WNIDs do not match task metadata")
    train_indices, holdout_indices = stratified_indices(
        indexing.targets, args.holdout_fraction, args.seed
    )
    train_base = ImageFolder(args.train_root, transform=train_transform)
    holdout_base = ImageFolder(args.train_root, transform=eval_transform)
    test_base = ImageFolder(args.test_root, transform=eval_transform)
    if set(test_base.classes) != set(mapping):
        raise ValueError("Test WNIDs do not match task metadata")
    train_dataset = torch.utils.data.Subset(
        RemappedDataset(train_base, mapping), train_indices
    )
    holdout_dataset = torch.utils.data.Subset(
        RemappedDataset(holdout_base, mapping), holdout_indices
    )
    test_dataset = RemappedDataset(test_base, mapping)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = min(
        len(train_loader), args.max_steps if args.max_steps is not None else len(train_loader)
    )
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = max(1, args.epochs * optimizer_steps_per_epoch)
    warmup = max(1, int(0.05 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_accuracy = -1.0
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        for batch_index, (images, targets) in enumerate(train_loader):
            if args.max_steps is not None and batch_index >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                raw_loss = torch.nn.functional.cross_entropy(logits, targets)
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            loss_sum += float(raw_loss.detach()) * len(targets)
            seen += len(targets)
            boundary = (batch_index + 1) % accumulation == 0
            final_batch = batch_index + 1 == batches_per_epoch
            if boundary or final_batch:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
        holdout_loss, holdout_accuracy = evaluate(model, holdout_loader, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "holdout_loss": holdout_loss,
            "holdout_accuracy": holdout_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "elapsed_minutes": (time.perf_counter() - started) / 60,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if holdout_accuracy > best_accuracy:
            best_accuracy = holdout_accuracy
            torch.save({
                "model": args.model,
                "num_classes": num_classes,
                "class_to_idx": mapping,
                "state_dict": model.state_dict(),
                "trained_num_heads": original_heads,
                "seed": args.seed,
                "task": args.task,
                "task_metadata": str(args.task_metadata),
                "training_protocol": vars(args) | {
                    "train_root": str(args.train_root),
                    "test_root": str(args.test_root),
                    "checkpoint": str(args.checkpoint),
                    "output": str(args.output),
                },
                "history": history,
            }, args.output)
    summary = {
        "best_holdout_accuracy": best_accuracy,
        "history": history,
        "train_examples": len(train_dataset),
        "holdout_examples": len(holdout_dataset),
        "test_examples": len(test_dataset),
        "task": args.task,
        "num_classes": num_classes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "scientific_scope": "paired-image semantic-label-granularity control",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"saved={args.output} best_holdout_accuracy={best_accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
