"""Phase-A pre-head predictor for the best representative head."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        choices=("hard", "listwise", "regression"),
        default="hard",
    )
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--score-clip", type=float, default=1.0)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument(
        "--max-train-images",
        type=int,
        help="Optional learning-curve cap applied before flattening.",
    )
    parser.add_argument("--train-layer-index", type=int)
    parser.add_argument("--val-layer-index", type=int)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--test-layer-index", type=int)
    parser.add_argument("--train-feature-file", type=Path)
    parser.add_argument("--val-feature-file", type=Path)
    parser.add_argument("--test-feature-file", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_shards(directory: Path) -> dict[str, np.ndarray]:
    files = sorted(directory.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"No shards found in {directory}")
    keys = (
        "dataset_index",
        "prehead",
        "taylor",
        "contribution_l2",
        "substitution",
        "single_drop",
    )
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for file in files:
        archive = np.load(file)
        for key in keys:
            pieces[key].append(archive[key])
    return {key: np.concatenate(values, axis=0) for key, values in pieces.items()}


def score_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    drop = np.maximum(data["single_drop"][..., None], 1e-12)
    scores = 1.0 - data["substitution"] / drop
    heads = scores.shape[-1]
    diagonal = np.eye(heads, dtype=bool)[None, None, :, :]
    return np.where(diagonal, np.nan, scores)


def select_layer(
    data: dict[str, np.ndarray],
    layer_index: int | None,
) -> dict[str, np.ndarray]:
    if layer_index is None:
        return data
    layer_keys = {
        "prehead",
        "taylor",
        "contribution_l2",
        "substitution",
        "single_drop",
    }
    return {
        key: (
            value[:, layer_index : layer_index + 1]
            if key in layer_keys
            else value
        )
        for key, value in data.items()
    }


def override_features(
    data: dict[str, np.ndarray],
    feature_file: Path | None,
) -> dict[str, np.ndarray]:
    if feature_file is None:
        return data
    features = np.load(feature_file)["features"]
    if len(features) != len(data["dataset_index"]):
        raise ValueError(
            f"Feature count {len(features)} != image count "
            f"{len(data['dataset_index'])}"
        )
    result = dict(data)
    result["prehead"] = features[:, None, :]
    return result


def flatten_examples(
    data: dict[str, np.ndarray],
    minimum_drop: float,
) -> dict[str, np.ndarray]:
    features = data["prehead"].astype(np.float32)
    samples, layers, feature_dim = features.shape
    heads = data["single_drop"].shape[-1]
    scores = score_matrix(data)

    sample_index, layer_index, source_index = np.where(
        data["single_drop"] > minimum_drop
    )
    candidate_scores = scores[sample_index, layer_index, source_index]
    labels = np.nanargmax(candidate_scores, axis=-1)
    return {
        "features": features[sample_index, layer_index],
        "layer": layer_index.astype(np.int64),
        "source": source_index.astype(np.int64),
        "label": labels.astype(np.int64),
        "scores": candidate_scores.astype(np.float32),
        "sample": sample_index.astype(np.int64),
        "taylor": data["taylor"][sample_index, layer_index].astype(np.float32),
        "l2": data["contribution_l2"][
            sample_index, layer_index, source_index
        ].astype(np.float32),
        "feature_dim": np.asarray(feature_dim),
        "layers_count": np.asarray(layers),
        "heads_count": np.asarray(heads),
    }


class RepresentativePredictor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        layers: int,
        heads: int,
        hidden: int,
        embedding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_embedding = nn.Embedding(layers, embedding)
        self.source_embedding = nn.Embedding(heads, embedding)
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 2 * embedding, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, heads),
        )

    def forward(
        self,
        feature: torch.Tensor,
        layer: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.cat(
            [
                feature,
                self.layer_embedding(layer),
                self.source_embedding(source),
            ],
            dim=-1,
        )
        logits = self.network(value)
        return logits.scatter(
            1,
            source[:, None],
            torch.full(
                (len(source), 1),
                -torch.inf,
                device=logits.device,
                dtype=logits.dtype,
            ),
        )


def policy_metrics(
    name: str,
    representatives: np.ndarray | None,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | str]:
    if representatives is None:
        selected = np.nanmean(scores, axis=1)
        accuracy = 1.0 / (scores.shape[1] - 1)
    else:
        selected = scores[np.arange(len(scores)), representatives]
        accuracy = float(np.mean(representatives == labels))
    return {
        "policy": name,
        "mean_recovery": float(np.mean(selected)),
        "median_recovery": float(np.median(selected)),
        "representative_accuracy": accuracy,
    }


def evaluate(
    model: nn.Module,
    examples: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    static_policy: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    features = (examples["features"] - mean) / std
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(examples["layer"]),
        torch.from_numpy(examples["source"]),
    )
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for feature, layer, source in loader:
            logits = model(
                feature.to(device),
                layer.to(device),
                source.to(device),
            )
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    predicted = np.concatenate(predictions)

    source = examples["source"]
    candidates = np.arange(int(examples["heads_count"]))[None, :]
    valid = candidates != source[:, None]
    l2 = np.where(valid, examples["l2"], np.inf)
    taylor = np.where(valid, examples["taylor"], -np.inf)
    static = static_policy[examples["layer"], source]

    rows = [
        policy_metrics("random_expected", None, examples["scores"], examples["label"]),
        policy_metrics(
            "current_l2",
            l2.argmin(axis=1),
            examples["scores"],
            examples["label"],
        ),
        policy_metrics(
            "highest_taylor_importance",
            taylor.argmax(axis=1),
            examples["scores"],
            examples["label"],
        ),
        policy_metrics(
            "static_train_representative",
            static,
            examples["scores"],
            examples["label"],
        ),
        policy_metrics(
            "prehead_predictor",
            predicted,
            examples["scores"],
            examples["label"],
        ),
        policy_metrics(
            "functional_oracle",
            examples["label"],
            examples["scores"],
            examples["label"],
        ),
    ]
    top2_accuracy = np.nan
    return rows, predicted


def static_representatives(
    examples: dict[str, np.ndarray],
    layers: int,
    heads: int,
) -> np.ndarray:
    policy = np.zeros((layers, heads), dtype=np.int64)
    for layer in range(layers):
        for source in range(heads):
            selector = (
                (examples["layer"] == layer)
                & (examples["source"] == source)
            )
            representatives = np.asarray(
                [head for head in range(heads) if head != source]
            )
            mean_score = np.mean(
                examples["scores"][selector][:, representatives],
                axis=0,
            )
            policy[layer, source] = int(
                representatives[np.argmax(mean_score)]
            )
    return policy


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_shards(args.train_dir)
    train_raw = select_layer(train_raw, args.train_layer_index)
    train_raw = override_features(train_raw, args.train_feature_file)
    if args.max_train_images is not None:
        total_images = len(train_raw["dataset_index"])
        selected = np.linspace(
            0,
            total_images - 1,
            num=min(args.max_train_images, total_images),
            dtype=int,
        )
        train_raw = {
            key: value[selected]
            for key, value in train_raw.items()
        }
    train = flatten_examples(train_raw, args.minimum_drop_kl)
    validation_raw = override_features(
        select_layer(
            load_shards(args.val_dir),
            args.val_layer_index,
        ),
        args.val_feature_file,
    )
    validation = flatten_examples(
        validation_raw,
        args.minimum_drop_kl,
    )
    feature_mean = train["features"].mean(axis=0)
    feature_std = train["features"].std(axis=0).clip(min=1e-5)

    device = torch.device(args.device)
    model = RepresentativePredictor(
        feature_dim=int(train["feature_dim"]),
        layers=int(train["layers_count"]),
        heads=int(train["heads_count"]),
        hidden=args.hidden_dim,
        embedding=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    normalized = (train["features"] - feature_mean) / feature_std
    train_set = TensorDataset(
        torch.from_numpy(normalized),
        torch.from_numpy(train["layer"]),
        torch.from_numpy(train["source"]),
        torch.from_numpy(train["label"]),
        torch.from_numpy(train["scores"]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    static_policy = static_representatives(
        train,
        int(train["layers_count"]),
        int(train["heads_count"]),
    )

    history = []
    best_recovery = -np.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for feature, layer, source, label, scores in loader:
            optimizer.zero_grad(set_to_none=True)
            source_device = source.to(device)
            logits = model(
                feature.to(device),
                layer.to(device),
                source_device,
            )
            if args.loss == "hard":
                loss = criterion(logits, label.to(device))
            else:
                target = scores.to(device).clamp(
                    min=-args.score_clip,
                    max=args.score_clip,
                )
                valid = torch.ones_like(target, dtype=torch.bool)
                valid.scatter_(1, source_device[:, None], False)
                if args.loss == "listwise":
                    target_logits = (
                        target / args.target_temperature
                    ).masked_fill(~valid, -torch.inf)
                    target_distribution = target_logits.softmax(dim=-1)
                    loss = -(
                        target_distribution
                        * logits.log_softmax(dim=-1)
                    )[valid].sum() / len(target)
                else:
                    loss = ((logits[valid] - target[valid]) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, _ = evaluate(
            model,
            validation,
            feature_mean,
            feature_std,
            device,
            static_policy,
        )
        predictor = next(
            row for row in metrics if row["policy"] == "prehead_predictor"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"val_{key}": value for key, value in predictor.items() if key != "policy"},
            }
        )
        if predictor["mean_recovery"] > best_recovery:
            best_recovery = float(predictor["mean_recovery"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_S={predictor['mean_recovery']:.4f} "
            f"val_acc={predictor['representative_accuracy']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    validation_metrics, validation_predictions = evaluate(
        model,
        validation,
        feature_mean,
        feature_std,
        device,
        static_policy,
    )
    pd.DataFrame(validation_metrics).to_csv(
        args.output_dir / "validation_policy_summary.csv",
        index=False,
    )
    if args.test_dir is not None:
        test = flatten_examples(
            override_features(
                select_layer(
                    load_shards(args.test_dir),
                    args.test_layer_index,
                ),
                args.test_feature_file,
            ),
            args.minimum_drop_kl,
        )
        metrics, predictions = evaluate(
            model,
            test,
            feature_mean,
            feature_std,
            device,
            static_policy,
        )
        reported = test
    else:
        metrics, predictions = validation_metrics, validation_predictions
        reported = validation
    pd.DataFrame(metrics).to_csv(args.output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        prediction=predictions,
        label=reported["label"],
        sample=reported["sample"],
        layer=reported["layer"],
        source=reported["source"],
        scores=reported["scores"],
    )
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "static_policy": static_policy,
            "args": vars(args),
        },
        args.output_dir / "model.pt",
    )
    metadata = {
        "train_images": int(len(np.unique(train["sample"]))),
        "validation_images": int(len(np.unique(validation["sample"]))),
        "train_examples": len(train["label"]),
        "validation_examples": len(validation["label"]),
        "test_images": (
            int(len(np.unique(reported["sample"])))
            if args.test_dir is not None
            else None
        ),
        "test_examples": (
            len(reported["label"]) if args.test_dir is not None else None
        ),
        "best_validation_recovery": best_recovery,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
