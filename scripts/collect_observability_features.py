"""Collect an information ladder for functional-substitution observability."""

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
from timm.data import ImageNetInfo, create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

from scripts.run_experiment0 import configure_cache
from src.head_intervention import clear_interventions, make_intervenable
from src.model_loading import create_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def shard_paths(directory: Path) -> list[str]:
    paths: list[str] = []
    for file in sorted(directory.glob("shard_*.npz")):
        paths.extend(np.load(file)["path"].tolist())
    if not paths:
        raise FileNotFoundError(f"No oracle shards found in {directory}")
    return paths


class PathDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        transform,
        synset_to_index: dict[str, int],
    ) -> None:
        self.paths = paths
        self.transform = transform
        self.synset_to_index = synset_to_index

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            value = self.transform(image.convert("RGB"))
        synset = Path(self.paths[index]).parent.name
        return value, int(self.synset_to_index[synset])


def residual_summary(hidden: torch.Tensor) -> torch.Tensor:
    patches = hidden[:, 1:].float()
    return torch.cat(
        [
            hidden[:, 0].float(),
            patches.mean(dim=1),
            patches.var(dim=1, unbiased=False),
        ],
        dim=-1,
    )


def split_qkv(
    attention,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, count, _ = tokens.shape
    qkv = (
        attention.qkv(tokens)
        .reshape(
            batch,
            count,
            3,
            attention.num_heads,
            attention.head_dim,
        )
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)
    return (
        attention.q_norm(query).float(),
        attention.k_norm(key).float(),
        value.float(),
    )


def pooled_qk_features(attention, hidden: torch.Tensor) -> torch.Tensor:
    """L1: Q/K vectors for CLS and the mean patch token."""
    summary_tokens = torch.stack(
        [hidden[:, 0], hidden[:, 1:].mean(dim=1)],
        dim=1,
    )
    query, key, _ = split_qkv(attention, summary_tokens)
    return torch.cat(
        [
            query[:, :, 0],
            key[:, :, 0],
            query[:, :, 1],
            key[:, :, 1],
        ],
        dim=-1,
    )


def token_qk_features(
    attention,
    hidden: torch.Tensor,
    level1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """L2: token-wise Q/K first and second moments."""
    query, key, value = split_qkv(attention, hidden)
    query_mean = query.mean(dim=2)
    key_mean = key.mean(dim=2)
    query_std = query.std(dim=2, unbiased=False)
    key_std = key.std(dim=2, unbiased=False)
    query_norm = query.norm(dim=-1)
    key_norm = key.norm(dim=-1)
    scalars = torch.stack(
        [
            query_norm.mean(dim=2),
            query_norm.std(dim=2, unbiased=False),
            key_norm.mean(dim=2),
            key_norm.std(dim=2, unbiased=False),
        ],
        dim=-1,
    )
    level2 = torch.cat(
        [
            level1,
            query_mean,
            key_mean,
            query_std,
            key_std,
            scalars,
        ],
        dim=-1,
    )
    return level2, query, key, value


def attention_features(
    attention,
    level2: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """L3: statistics that require materializing the attention logits."""
    logits = (query * attention.scale) @ key.transpose(-2, -1)
    probabilities = logits.softmax(dim=-1)
    entropy = -(
        probabilities.clamp_min(1e-12)
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1) / np.log(probabilities.shape[-1])
    maximum = probabilities.max(dim=-1).values
    diagonal = probabilities.diagonal(dim1=-2, dim2=-1)
    scalar = torch.stack(
        [
            entropy.mean(dim=-1),
            entropy.std(dim=-1, unbiased=False),
            entropy[:, :, 0],
            maximum.mean(dim=-1),
            maximum.std(dim=-1, unbiased=False),
            maximum[:, :, 0],
            diagonal.mean(dim=-1),
            probabilities[:, :, 0, 0],
            logits.mean(dim=(-2, -1)),
            logits.std(dim=(-2, -1), unbiased=False),
        ],
        dim=-1,
    )
    return torch.cat([level2, scalar], dim=-1)


def tensor_summary(value: torch.Tensor) -> torch.Tensor:
    """CLS, token mean/std, and norm moments for [B,H,N,D]."""
    norm = value.norm(dim=-1)
    return torch.cat(
        [
            value[:, :, 0],
            value.mean(dim=2),
            value.std(dim=2, unbiased=False),
            norm.mean(dim=2, keepdim=True),
            norm.std(dim=2, unbiased=False, keepdim=True),
        ],
        dim=-1,
    )


def post_wo_contributions(attention, heads: torch.Tensor) -> torch.Tensor:
    """Map each [B,H,N,D] head independently through its W_O slice."""
    contributions = []
    for head in range(attention.num_heads):
        begin = head * attention.head_dim
        end = begin + attention.head_dim
        contributions.append(
            F.linear(
                heads[:, head],
                attention.proj.weight[:, begin:end].float(),
                bias=None,
            )
        )
    return torch.stack(contributions, dim=1)


def gradient_features(
    model: torch.nn.Module,
    wrappers,
    wrapper,
    image: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Signed/absolute CE and predicted-logit gate gradients per head."""
    clear_interventions(wrappers)
    scales = torch.ones(
        len(image),
        wrapper.num_heads,
        device=image.device,
        dtype=image.dtype,
        requires_grad=True,
    )
    wrapper.set_head_scales(scales)
    logits = model(image)
    loss = F.cross_entropy(logits.float(), target, reduction="sum")
    gradient_loss = torch.autograd.grad(
        loss,
        scales,
        retain_graph=True,
    )[0]
    predicted = logits.argmax(dim=-1)
    selected_logit = logits.float().gather(1, predicted[:, None]).sum()
    gradient_logit = torch.autograd.grad(selected_logit, scales)[0]
    clear_interventions(wrappers)
    return torch.stack(
        [
            gradient_loss,
            gradient_loss.abs(),
            gradient_logit,
            gradient_logit.abs(),
        ],
        dim=-1,
    ).float()


def main() -> None:
    args = parse_args()
    configure_cache()
    device = torch.device("cuda")
    loaded_model, custom_class_to_idx = create_timm_model(
        args.model,
        args.checkpoint,
    )
    model = (
        loaded_model
        .eval()
        .to(device=device, dtype=torch.float16)
    )
    wrappers = make_intervenable(model)
    if args.layer < 0 or args.layer >= len(wrappers):
        raise ValueError(f"Invalid layer {args.layer}")
    wrapper = wrappers[args.layer]
    attention = wrapper.attention

    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )
    paths = shard_paths(args.oracle_dir)
    if custom_class_to_idx is None:
        labels = ImageNetInfo().label_names()
        synset_to_index = {
            synset: index for index, synset in enumerate(labels)
        }
    else:
        synset_to_index = custom_class_to_idx
    loader = DataLoader(
        PathDataset(paths, transform, synset_to_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    records: dict[str, list[np.ndarray]] = {
        "global": [],
        "level1": [],
        "level2": [],
        "level3": [],
        "level4": [],
        "level5": [],
        "level6": [],
        "level7": [],
    }
    for batch_index, (image, target) in enumerate(loader):
        image = image.to(device=device, dtype=torch.float16)
        target = target.to(device=device)
        clear_interventions(wrappers)
        wrapper.capture_input = True
        with torch.inference_mode():
            model(image)
            hidden = wrapper.last_input
            if hidden is None:
                raise RuntimeError("Pre-head input capture failed.")
            global_feature = residual_summary(hidden)
            level1 = pooled_qk_features(attention, hidden)
            level2, query, key, value = token_qk_features(
                attention,
                hidden,
                level1,
            )
            level3 = attention_features(
                attention,
                level2,
                query,
                key,
            )
            value_feature = tensor_summary(value)
            level4 = torch.cat([level3, value_feature], dim=-1)
            probabilities = (
                (query * attention.scale) @ key.transpose(-2, -1)
            ).softmax(dim=-1)
            aggregated = probabilities @ value
            level5 = torch.cat(
                [level4, tensor_summary(aggregated)],
                dim=-1,
            )
            contribution = post_wo_contributions(attention, aggregated)
            level6 = torch.cat(
                [level5, tensor_summary(contribution)],
                dim=-1,
            )
        gradient = gradient_features(
            model,
            wrappers,
            wrapper,
            image,
            target,
        )
        level7 = torch.cat([level6, gradient], dim=-1)
        for name, value in (
            ("global", global_feature),
            ("level1", level1),
            ("level2", level2),
            ("level3", level3),
            ("level4", level4),
            ("level5", level5),
            ("level6", level6),
            ("level7", level7),
        ):
            records[name].append(value.to(torch.float16).cpu().numpy())
        print(f"batch={batch_index + 1}/{len(loader)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        global_features=np.concatenate(records["global"], axis=0),
        level1=np.concatenate(records["level1"], axis=0),
        level2=np.concatenate(records["level2"], axis=0),
        level3=np.concatenate(records["level3"], axis=0),
        level4=np.concatenate(records["level4"], axis=0),
        level5=np.concatenate(records["level5"], axis=0),
        level6=np.concatenate(records["level6"], axis=0),
        level7=np.concatenate(records["level7"], axis=0),
        paths=np.asarray(paths),
        layer=args.layer,
        definitions=np.asarray(
            [
                "L0: CLS + patch mean + patch variance",
                "L1: L0 + pooled-token Q/K vectors",
                "L2: L1 + token-wise Q/K moments",
                "L3: L2 + full attention-logit statistics",
                "L4: L3 + V content statistics",
                "L5: L4 + aggregated A@V head-output statistics",
                "L6: L5 + post-WO contribution statistics",
                "L7: L6 + dense-model gate-gradient statistics",
            ]
        ),
    )
    os.replace(temporary, args.output)
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
