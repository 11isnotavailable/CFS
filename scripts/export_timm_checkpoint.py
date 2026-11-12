"""Export a timm pretrained model into the project's offline checkpoint format."""

from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = timm.create_model(args.model, pretrained=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "num_classes": int(model.num_classes),
            "class_to_idx": None,
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    parameters = sum(value.numel() for value in model.state_dict().values())
    print(f"output={args.output.resolve()} parameters={parameters}", flush=True)


if __name__ == "__main__":
    main()
