"""Train an explicit pair scorer S_hat(x, source, representative)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_representative_predictor import load_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--train-layer-index", type=int, default=0)
    parser.add_argument("--val-layer-index", type=int, default=0)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-layer-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--level",
        choices=(
            "level0",
            "level1",
            "level2",
            "level3",
            "level4",
            "level5",
            "level6",
            "level7",
            "value_only",
            "aggregated_only",
            "contribution_only",
            "gradient_only",
            "gradient_ce_only",
            "gradient_logit_only",
            "gradient_abs_ce_only",
        ),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--head-projection-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--score-clip", type=float, default=1.0)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_examples(
    oracle_dir: Path,
    feature_file: Path,
    level: str,
    minimum_drop: float,
    layer_index: int,
) -> dict[str, np.ndarray | int]:
    oracle = load_shards(oracle_dir)
    archive = np.load(feature_file)
    if len(archive["paths"]) != len(oracle["dataset_index"]):
        raise ValueError("Feature/oracle image count mismatch.")
    global_features = archive["global_features"].astype(np.float32)
    if level == "level0":
        head_features = None
    elif level == "value_only":
        begin = archive["level3"].shape[-1]
        head_features = archive["level4"][..., begin:].astype(np.float32)
    elif level == "aggregated_only":
        begin = archive["level4"].shape[-1]
        head_features = archive["level5"][..., begin:].astype(np.float32)
    elif level == "contribution_only":
        begin = archive["level5"].shape[-1]
        head_features = archive["level6"][..., begin:].astype(np.float32)
    elif level == "gradient_only":
        begin = archive["level6"].shape[-1]
        head_features = archive["level7"][..., begin:].astype(np.float32)
    elif level == "gradient_ce_only":
        begin = archive["level6"].shape[-1]
        head_features = archive["level7"][..., begin : begin + 2].astype(
            np.float32
        )
    elif level == "gradient_logit_only":
        begin = archive["level6"].shape[-1]
        head_features = archive["level7"][..., begin + 2 : begin + 4].astype(
            np.float32
        )
    elif level == "gradient_abs_ce_only":
        begin = archive["level6"].shape[-1]
        head_features = archive["level7"][..., begin + 1 : begin + 2].astype(
            np.float32
        )
    else:
        head_features = archive[level].astype(np.float32)
    substitution = oracle["substitution"][:, layer_index].astype(np.float32)
    drops = oracle["single_drop"][:, layer_index].astype(np.float32)
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    valid_source = drops > minimum_drop
    sample, source = np.where(valid_source)
    gradient_begin = archive["level6"].shape[-1]
    gradient_features = archive["level7"][
        ..., gradient_begin : gradient_begin + 4
    ].astype(np.float32)
    return {
        "global": global_features[sample],
        "head_images": head_features,
        "scores": scores[sample, source],
        "source": source.astype(np.int64),
        "sample": sample.astype(np.int64),
        "taylor": oracle["taylor"][sample, layer_index].astype(np.float32),
        "l2": oracle["contribution_l2"][
            sample, layer_index, source
        ].astype(np.float32),
        "heads": heads,
        "gradient_ce_signed": gradient_features[..., 0],
        "gradient_logit_signed": gradient_features[..., 2],
    }


class PairScorePredictor(nn.Module):
    def __init__(
        self,
        global_dim: int,
        head_dim: int,
        heads: int,
        hidden: int,
        head_hidden: int,
        embedding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.GELU(),
        )
        self.head_encoder = (
            nn.Sequential(nn.Linear(head_dim, head_hidden), nn.GELU())
            if head_dim > 0
            else None
        )
        self.source_embedding = nn.Embedding(heads, embedding)
        self.candidate_embedding = nn.Embedding(heads, embedding)
        pair_dim = hidden + 2 * embedding
        if self.head_encoder is not None:
            pair_dim += 4 * head_hidden
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        head_feature: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(source)
        candidates = torch.arange(
            self.heads,
            device=global_feature.device,
        )
        global_encoded = self.global_encoder(global_feature)
        pieces = [
            global_encoded[:, None, :].expand(-1, self.heads, -1),
            self.source_embedding(source)[:, None, :].expand(
                -1, self.heads, -1
            ),
            self.candidate_embedding(candidates)[None, :, :].expand(
                batch, -1, -1
            ),
        ]
        if self.head_encoder is not None:
            encoded = self.head_encoder(head_feature)
            source_feature = encoded[
                torch.arange(batch, device=source.device),
                source,
            ]
            source_expanded = source_feature[:, None, :].expand(
                -1, self.heads, -1
            )
            pieces.extend(
                [
                    source_expanded,
                    encoded,
                    (source_expanded - encoded).abs(),
                    source_expanded * encoded,
                ]
            )
        logits = self.scorer(torch.cat(pieces, dim=-1)).squeeze(-1)
        return logits.scatter(
            1,
            source[:, None],
            torch.full(
                (batch, 1),
                -torch.inf,
                device=logits.device,
            ),
        )


def normalization(
    train: dict,
    projection_dim: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int,
]:
    global_mean = train["global"].mean(axis=0)
    global_std = train["global"].std(axis=0).clip(min=1e-5)
    if train["head_images"] is None:
        return (
            global_mean,
            global_std,
            None,
            None,
            None,
            projection_dim,
        )
    head_mean = train["head_images"].mean(axis=(0, 1))
    head_std = train["head_images"].std(axis=(0, 1)).clip(min=1e-5)
    input_dim = train["head_images"].shape[-1]
    if input_dim == projection_dim:
        projection = np.eye(input_dim, dtype=np.float32)
    else:
        rng = np.random.default_rng(20260731 + input_dim)
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(input_dim),
            size=(input_dim, projection_dim),
        ).astype(np.float32)
    return (
        global_mean,
        global_std,
        head_mean,
        head_std,
        projection,
        projection_dim,
    )


def normalized_tensors(
    examples: dict,
    statistics: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        global_mean,
        global_std,
        head_mean,
        head_std,
        projection,
        projection_dim,
    ) = statistics
    global_feature = (
        examples["global"] - global_mean
    ) / global_std
    if examples["head_images"] is None:
        head_feature = np.zeros(
            (
                len(global_feature),
                int(examples["heads"]),
                projection_dim,
            ),
            dtype=np.float32,
        )
    else:
        normalized_images = (
            examples["head_images"] - head_mean[None, None, :]
        ) / head_std[None, None, :]
        projected_images = normalized_images @ projection
        head_feature = projected_images[examples["sample"]]
    return torch.from_numpy(global_feature), torch.from_numpy(head_feature)


def realized_scores(
    examples: dict,
    representatives: np.ndarray | None,
) -> np.ndarray:
    if representatives is None:
        return np.nanmean(examples["scores"], axis=1)
    return examples["scores"][
        np.arange(len(representatives)),
        representatives,
    ]


def evaluate(
    model: nn.Module,
    examples: dict,
    statistics: tuple,
    device: torch.device,
    static_policy: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    global_tensor, head_tensor = normalized_tensors(examples, statistics)
    loader = DataLoader(
        TensorDataset(
            global_tensor,
            head_tensor,
            torch.from_numpy(examples["source"]),
        ),
        batch_size=1024,
        shuffle=False,
    )
    prediction_scores = []
    model.eval()
    with torch.inference_mode():
        for global_feature, head_feature, source in loader:
            logits = model(
                global_feature.to(device),
                head_feature.to(device),
                source.to(device),
            )
            prediction_scores.append(logits.cpu().numpy())
    predicted_scores = np.concatenate(prediction_scores)
    predicted = predicted_scores.argmax(axis=-1)
    heads = int(examples["heads"])
    valid = np.arange(heads)[None, :] != examples["source"][:, None]
    alpha = np.linspace(0.0, 3.0, 31, dtype=np.float32)

    def first_order_representative(key: str) -> np.ndarray:
        gradient = examples[key][examples["sample"]]
        source_gradient = gradient[
            np.arange(len(examples["source"])),
            examples["source"],
        ]
        residual = np.abs(
            -source_gradient[:, None, None]
            + gradient[:, :, None] * (alpha[None, None, :] - 1.0)
        )
        cost = residual.min(axis=-1)
        return np.where(valid, cost, np.inf).argmin(axis=1)

    choices = {
        "random_expected": None,
        "current_l2": np.where(valid, examples["l2"], np.inf).argmin(axis=1),
        "highest_taylor_importance": np.where(
            valid, examples["taylor"], -np.inf
        ).argmax(axis=1),
        "signed_ce_first_order": first_order_representative(
            "gradient_ce_signed"
        ),
        "signed_logit_first_order": first_order_representative(
            "gradient_logit_signed"
        ),
        "static_train_representative": static_policy[examples["source"]],
        "pair_score_predictor": predicted,
        "functional_oracle": np.nanargmax(examples["scores"], axis=1),
    }
    oracle = choices["functional_oracle"]
    rows = []
    for name, representatives in choices.items():
        values = realized_scores(examples, representatives)
        accuracy = (
            1.0 / (heads - 1)
            if representatives is None
            else float(np.mean(representatives == oracle))
        )
        row = {
            "policy": name,
            "mean_recovery": float(values.mean()),
            "median_recovery": float(np.median(values)),
            "representative_accuracy": accuracy,
            "pair_spearman_mean": np.nan,
            "pair_spearman_median": np.nan,
        }
        if name == "pair_score_predictor":
            correlations = []
            for index, source in enumerate(examples["source"]):
                valid_candidates = np.arange(heads) != source
                predicted_row = predicted_scores[index, valid_candidates]
                target_row = examples["scores"][index, valid_candidates]
                if np.std(predicted_row) == 0 or np.std(target_row) == 0:
                    continue
                correlation = spearmanr(predicted_row, target_row).statistic
                if np.isfinite(correlation):
                    correlations.append(correlation)
            row["pair_spearman_mean"] = float(np.mean(correlations))
            row["pair_spearman_median"] = float(np.median(correlations))
        rows.append(row)
    return pd.DataFrame(rows), predicted, predicted_scores


def static_policy(train: dict) -> np.ndarray:
    heads = int(train["heads"])
    policy = np.zeros(heads, dtype=np.int64)
    for source in range(heads):
        selector = train["source"] == source
        representatives = np.asarray(
            [head for head in range(heads) if head != source]
        )
        means = train["scores"][selector][:, representatives].mean(axis=0)
        policy[source] = int(representatives[np.argmax(means)])
    return policy


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_examples(
        args.train_dir,
        args.train_features,
        args.level,
        args.minimum_drop_kl,
        args.train_layer_index,
    )
    validation = load_examples(
        args.val_dir,
        args.val_features,
        args.level,
        args.minimum_drop_kl,
        args.val_layer_index,
    )
    statistics = normalization(train, args.head_projection_dim)
    global_tensor, head_tensor = normalized_tensors(train, statistics)
    train_set = TensorDataset(
        global_tensor,
        head_tensor,
        torch.from_numpy(train["source"]),
        torch.from_numpy(train["scores"]),
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    device = torch.device(args.device)
    head_dim = args.head_projection_dim
    model = PairScorePredictor(
        global_dim=train["global"].shape[-1],
        head_dim=head_dim,
        heads=int(train["heads"]),
        hidden=args.hidden_dim,
        head_hidden=args.head_hidden_dim,
        embedding=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    fixed = static_policy(train)
    history = []
    best_recovery = -np.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for global_feature, head_feature, source, scores in loader:
            optimizer.zero_grad(set_to_none=True)
            source = source.to(device)
            logits = model(
                global_feature.to(device),
                head_feature.to(device),
                source,
            )
            target = scores.to(device).clamp(
                -args.score_clip,
                args.score_clip,
            )
            valid = torch.ones_like(target, dtype=torch.bool)
            valid.scatter_(1, source[:, None], False)
            target_logits = (
                target / args.target_temperature
            ).masked_fill(~valid, -torch.inf)
            distribution = target_logits.softmax(dim=-1)
            loss = -(
                distribution * logits.log_softmax(dim=-1)
            )[valid].sum() / len(target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        metrics, _, _ = evaluate(
            model,
            validation,
            statistics,
            device,
            fixed,
        )
        predictor = metrics.set_index("policy").loc["pair_score_predictor"]
        recovery = float(predictor["mean_recovery"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_recovery": recovery,
                "val_accuracy": float(predictor["representative_accuracy"]),
            }
        )
        if recovery > best_recovery:
            best_recovery = recovery
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_S={recovery:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No checkpoint produced.")
    model.load_state_dict(best_state)
    validation_metrics, validation_prediction, validation_scores = evaluate(
        model,
        validation,
        statistics,
        device,
        fixed,
    )
    validation_metrics.to_csv(
        args.output_dir / "validation_policy_summary.csv",
        index=False,
    )
    if args.test_dir is not None:
        if args.test_features is None:
            raise ValueError("--test-features is required with --test-dir")
        reported = load_examples(
            args.test_dir,
            args.test_features,
            args.level,
            args.minimum_drop_kl,
            args.test_layer_index,
        )
        metrics, prediction, prediction_scores = evaluate(
            model,
            reported,
            statistics,
            device,
            fixed,
        )
    else:
        reported = validation
        metrics = validation_metrics
        prediction = validation_prediction
        prediction_scores = validation_scores
    metrics.to_csv(args.output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(history).to_csv(
        args.output_dir / "training_history.csv",
        index=False,
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        prediction=prediction,
        prediction_scores=prediction_scores,
        scores=reported["scores"],
        source=reported["source"],
        sample=reported["sample"],
    )
    torch.save(
        {
            "state_dict": best_state,
            "statistics": statistics,
            "static_policy": fixed,
            "args": vars(args),
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "level": args.level,
                "train_images": len(np.unique(train["sample"])),
                "validation_images": len(np.unique(validation["sample"])),
                "train_groups": len(train["source"]),
                "validation_groups": len(validation["source"]),
                "test_images": (
                    len(np.unique(reported["sample"]))
                    if args.test_dir is not None
                    else None
                ),
                "test_groups": (
                    len(reported["source"])
                    if args.test_dir is not None
                    else None
                ),
                "global_dim": train["global"].shape[-1],
                "head_dim": head_dim,
                "raw_head_dim": (
                    0
                    if train["head_images"] is None
                    else train["head_images"].shape[-1]
                ),
                "best_validation_recovery": best_recovery,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
