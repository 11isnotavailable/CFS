"""Fine-tune BERT-base on SST-2 with an internal, train-only validation split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--internal-val-size", type=int, default=2048)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def configure_cache() -> None:
    root = PROJECT_ROOT / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "datasets"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            prediction = model(**inputs).logits.argmax(dim=-1)
            correct += int((prediction == labels).sum())
            total += len(labels)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    configure_cache()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    raw = load_dataset("nyu-mll/glue", "sst2")
    shuffled = raw["train"].shuffle(seed=args.seed)
    val = shuffled.select(range(args.internal_val_size))
    train = shuffled.select(range(args.internal_val_size, len(shuffled)))
    if args.train_limit is not None:
        train = train.select(range(min(args.train_limit, len(train))))

    def tokenize(batch):
        result = tokenizer(
            batch["sentence"], truncation=True, max_length=args.max_length
        )
        result["labels"] = batch["label"]
        return result

    remove = raw["train"].column_names
    train = train.map(tokenize, batched=True, remove_columns=remove)
    val = val.map(tokenize, batched=True, remove_columns=remove)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    ).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = examples = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(**inputs).logits
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(labels)
            examples += len(labels)
            if step % 250 == 0:
                print(
                    f"epoch={epoch + 1} step={step}/{len(train_loader)} "
                    f"loss={running_loss / examples:.4f}",
                    flush=True,
                )
        val_accuracy = accuracy(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / examples,
            "internal_val_accuracy": val_accuracy,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_accuracy > best:
            best = val_accuracy
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.model,
        "task": "glue/sst2",
        "seed": args.seed,
        "internal_validation_is_train_only": True,
        "internal_val_size": len(val),
        "train_size": len(train),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_internal_val_accuracy": best,
        "history": history,
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
