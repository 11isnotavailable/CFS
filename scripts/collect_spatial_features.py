"""Collect cheap spatial token sketches for an existing oracle dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

from scripts.run_experiment0 import configure_cache
from src.head_intervention import clear_interventions, make_intervenable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="deit_small_patch16_224")
    return parser.parse_args()


def shard_paths(directory: Path) -> list[str]:
    paths = []
    for file in sorted(directory.glob("shard_*.npz")):
        paths.extend(np.load(file)["path"].tolist())
    if not paths:
        raise FileNotFoundError(f"No oracle shards in {directory}")
    return paths


class PathDataset(Dataset):
    def __init__(self, paths: list[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def main() -> None:
    args = parse_args()
    configure_cache()
    import timm

    device = torch.device("cuda")
    model = (
        timm.create_model(args.model, pretrained=True)
        .eval()
        .to(device=device, dtype=torch.float16)
    )
    wrappers = make_intervenable(model)
    if args.layer < 0 or args.layer >= len(wrappers):
        raise ValueError(f"Invalid layer {args.layer}")
    wrapper = wrappers[args.layer]
    wrapper.capture_input = True

    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )
    paths = shard_paths(args.oracle_dir)
    loader = DataLoader(
        PathDataset(paths, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    records = []
    for batch_index, image in enumerate(loader):
        clear_interventions(wrappers)
        wrapper.capture_input = True
        with torch.inference_mode():
            model(image.to(device=device, dtype=torch.float16))
        hidden = wrapper.last_input
        if hidden is None:
            raise RuntimeError("Pre-head capture failed.")
        cls = hidden[:, :1]
        patches = hidden[:, 1:]
        side = int(round(patches.shape[1] ** 0.5))
        if side * side != patches.shape[1]:
            raise ValueError("Patch token count is not square.")
        spatial = patches.reshape(
            len(hidden), side, side, hidden.shape[-1]
        ).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(
            spatial.float(),
            (args.pool_size, args.pool_size),
        )
        pooled = pooled.flatten(2).transpose(1, 2).to(torch.float16)
        feature = torch.cat([cls, pooled], dim=1).flatten(1)
        records.append(feature.cpu().numpy())
        print(f"batch={batch_index + 1}/{len(loader)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=np.concatenate(records, axis=0),
        paths=np.asarray(paths),
        layer=args.layer,
        pool_size=args.pool_size,
    )
    os.replace(temporary, args.output)
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
