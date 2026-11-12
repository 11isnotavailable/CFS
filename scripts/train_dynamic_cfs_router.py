"""Train a budget-aware, forward-only CFS router on frozen ViT oracles."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.dynamic_cfs_router import (
    candidate_subset_utilities,
    DynamicCFSRouter,
    realized_utilities,
    selected_masks,
    subset_masks,
    subset_utilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--keep-counts", default="2,3,4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--subset-temperature", type=float, default=0.05)
    parser.add_argument("--score-weight", type=float, default=0.5)
    parser.add_argument("--importance-weight", type=float, default=0.5)
    parser.add_argument("--alpha-weight", type=float, default=0.1)
    parser.add_argument("--structured-weight", type=float, default=1.0)
    parser.add_argument("--joint-train-dir", type=Path)
    parser.add_argument("--joint-val-dir", type=Path)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--joint-temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def load_router_data(directory: Path, layer_index: int) -> dict[str, np.ndarray]:
    files = sorted(directory.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"No oracle shards found in {directory}")
    keys = (
        "dataset_index",
        "path",
        "prehead",
        "taylor",
        "contribution_l2",
        "contribution_norm",
        "substitution",
        "best_alpha",
        "single_drop",
    )
    pieces = {key: [] for key in keys}
    for file in files:
        archive = np.load(file)
        for key in keys:
            pieces[key].append(archive[key])
    data = {
        key: np.concatenate(values, axis=0) for key, values in pieces.items()
    }
    drops = data["single_drop"][:, layer_index].astype(np.float32)
    substitution = data["substitution"][:, layer_index].astype(np.float32)
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    diagonal = np.arange(heads)
    scores[:, diagonal, diagonal] = np.nan
    return {
        "global": data["prehead"][:, layer_index].astype(np.float32),
        "scores": scores,
        "drops": drops,
        "alpha": data["best_alpha"][:, layer_index].astype(np.float32),
        "taylor": data["taylor"][:, layer_index].astype(np.float32),
        "l2": data["contribution_l2"][:, layer_index].astype(np.float32),
        "norm": data["contribution_norm"][:, layer_index].astype(np.float32),
        "dataset_index": data["dataset_index"].astype(np.int64),
        "paths": data["path"],
    }


def tensors(data: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray):
    feature = (data["global"] - mean) / std
    scores = np.nan_to_num(data["scores"], nan=-1.0).clip(-1, 1)
    alpha = np.nan_to_num(data["alpha"], nan=1.0).clip(0, 3)
    return (
        torch.from_numpy(feature.astype(np.float32)),
        torch.from_numpy(scores.astype(np.float32)),
        torch.from_numpy(data["drops"].astype(np.float32)),
        torch.from_numpy(alpha.astype(np.float32)),
    )


def load_joint_teacher(directory: Path) -> dict[str, np.ndarray]:
    files = sorted(directory.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"No joint teacher shards found in {directory}")
    keys = ("dataset_index", "path", "candidate_masks", "joint_kl")
    parts = {key: [] for key in keys}
    for file in files:
        with np.load(file) as archive:
            for key in keys:
                parts[key].append(archive[key])
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def assert_joint_alignment(
    router_data: dict[str, np.ndarray], joint: dict[str, np.ndarray]
) -> None:
    if not np.array_equal(router_data["dataset_index"], joint["dataset_index"]):
        raise ValueError("Joint teacher dataset indices do not align")
    if not np.array_equal(router_data["paths"].astype(str), joint["path"].astype(str)):
        raise ValueError("Joint teacher paths do not align")


def loss_terms(
    predicted_scores: torch.Tensor,
    importance_logits: torch.Tensor,
    predicted_alpha: torch.Tensor,
    target_scores: torch.Tensor,
    drops: torch.Tensor,
    target_alpha: torch.Tensor,
    masks_by_budget: dict[int, torch.Tensor],
    target_temperature: float,
    subset_temperature: float,
    joint_masks: torch.Tensor | None = None,
    joint_kl: torch.Tensor | None = None,
    joint_temperature: float = 0.5,
) -> dict[str, torch.Tensor]:
    heads = target_scores.shape[-1]
    off_diagonal = ~torch.eye(heads, dtype=torch.bool, device=target_scores.device)
    valid = off_diagonal[None].expand_as(target_scores)
    target_logits = (target_scores / target_temperature).masked_fill(~valid, -torch.inf)
    predicted_logits = predicted_scores.masked_fill(~valid, -torch.inf)
    target_distribution = target_logits.softmax(dim=-1)
    listwise_terms = torch.where(
        valid,
        target_distribution * predicted_logits.log_softmax(dim=-1),
        torch.zeros_like(target_distribution),
    )
    listwise = -listwise_terms.sum()
    listwise = listwise / (len(target_scores) * heads)
    score_regression = F.smooth_l1_loss(
        predicted_scores[valid], target_scores[valid]
    )

    target_importance = drops.clamp_min(0)
    target_importance = target_importance / target_importance.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    importance = -(
        target_importance * importance_logits.log_softmax(dim=-1)
    ).sum(dim=-1).mean()

    alpha_weights = target_distribution.masked_fill(~valid, 0)
    alpha = (
        alpha_weights * F.smooth_l1_loss(
            predicted_alpha, target_alpha, reduction="none"
        )
    ).sum() / (len(target_scores) * heads)

    predicted_importance = importance_logits.softmax(dim=-1)
    structured_parts = []
    for masks in masks_by_budget.values():
        target_utility = subset_utilities(target_scores, drops, masks)
        predicted_utility = subset_utilities(
            predicted_scores, predicted_importance, masks
        )
        target_subset = (target_utility / subset_temperature).softmax(dim=-1)
        structured_parts.append(
            -(
                target_subset
                * (predicted_utility / subset_temperature).log_softmax(dim=-1)
            ).sum(dim=-1).mean()
        )
    structured = torch.stack(structured_parts).mean()
    result = {
        "listwise": listwise,
        "score": score_regression,
        "importance": importance,
        "alpha": alpha,
        "structured": structured,
    }
    if joint_masks is not None and joint_kl is not None:
        joint_parts = []
        for budget in range(joint_masks.shape[1]):
            predicted_joint = candidate_subset_utilities(
                predicted_scores,
                predicted_importance,
                joint_masks[:, budget],
            )
            error = joint_kl[:, budget]
            standardized = (error - error.mean(dim=-1, keepdim=True)) / error.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            target = (-standardized / joint_temperature).softmax(dim=-1)
            joint_parts.append(
                -(
                    target
                    * (predicted_joint / subset_temperature).log_softmax(dim=-1)
                ).sum(dim=-1).mean()
            )
        result["joint"] = torch.stack(joint_parts).mean()
    return result


def joint_validation_regret(
    model: DynamicCFSRouter,
    data: dict[str, np.ndarray],
    joint: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> float:
    feature, _, _, _ = tensors(data, mean, std)
    masks = torch.from_numpy(joint["candidate_masks"].astype(bool))
    errors = torch.from_numpy(joint["joint_kl"].astype(np.float32))
    regrets = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(feature), 128):
            stop = begin + 128
            score, importance, _ = model(feature[begin:stop].to(device))
            batch_masks = masks[begin:stop].to(device)
            batch_errors = errors[begin:stop].to(device)
            for budget in range(batch_masks.shape[1]):
                utility = candidate_subset_utilities(
                    score, importance.softmax(dim=-1), batch_masks[:, budget]
                )
                choice = utility.argmax(dim=-1)
                chosen = batch_errors[:, budget].gather(1, choice[:, None]).squeeze(1)
                regrets.append(chosen - batch_errors[:, budget].min(dim=-1).values)
    return float(torch.cat(regrets).mean())


def choose_top(values: torch.Tensor, keep: int) -> torch.Tensor:
    result = torch.zeros_like(values, dtype=torch.bool)
    result.scatter_(1, values.topk(keep, dim=-1).indices, True)
    return result


def similarity_scores(l2: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(l2.astype(np.float32))
    maximum = value.amax(dim=(-2, -1), keepdim=True)
    minimum = value.amin(dim=(-2, -1), keepdim=True)
    return 1.0 - (value - minimum) / (maximum - minimum).clamp_min(1e-8)


def evaluate(
    model: DynamicCFSRouter,
    data: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    keep_counts: list[int],
    static_masks: dict[int, torch.Tensor],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    feature, target_scores, drops, _ = tensors(data, mean, std)
    model.eval()
    predicted_score_parts, importance_parts, alpha_parts = [], [], []
    with torch.inference_mode():
        for begin in range(0, len(feature), 512):
            score, importance, alpha = model(feature[begin : begin + 512].to(device))
            predicted_score_parts.append(score.cpu())
            importance_parts.append(importance.cpu())
            alpha_parts.append(alpha.cpu())
    predicted_scores = torch.cat(predicted_score_parts)
    importance_logits = torch.cat(importance_parts)
    predicted_alpha = torch.cat(alpha_parts)
    heads = target_scores.shape[-1]
    uniform = torch.ones_like(drops)
    l2_scores = similarity_scores(data["l2"])
    rows = []
    choices: dict[str, np.ndarray] = {}
    for keep in keep_counts:
        masks = subset_masks(heads, keep)
        oracle_mask, oracle_prediction_utility = selected_masks(
            target_scores, drops, masks
        )
        router_mask, _ = selected_masks(
            predicted_scores, importance_logits.softmax(dim=-1), masks
        )
        similarity_mask, _ = selected_masks(l2_scores, uniform, masks)
        policy_masks = {
            "random_expected": None,
            "static_cfs": static_masks[keep].expand(len(drops), -1),
            "current_l2": similarity_mask,
            "magnitude": choose_top(torch.from_numpy(data["norm"]), keep),
            "taylor": choose_top(torch.from_numpy(data["taylor"]), keep),
            "deletion_oracle": choose_top(drops, keep),
            "dynamic_cfs_router": router_mask,
            "pairwise_cfs_oracle": oracle_mask,
        }
        all_utilities = subset_utilities(target_scores, drops, masks)
        for policy, keep_mask in policy_masks.items():
            if keep_mask is None:
                utility = all_utilities.mean(dim=-1)
                overlap = torch.full_like(utility, float(keep) / heads)
            else:
                utility = realized_utilities(
                    target_scores, drops, keep_mask
                )
                overlap = (
                    (keep_mask & oracle_mask).sum(dim=-1).float() / keep
                )
                choices[f"keep{keep}_{policy}"] = keep_mask.numpy()
            rows.append(
                {
                    "keep": keep,
                    "keep_fraction": keep / heads,
                    "policy": policy,
                    "pairwise_surrogate_utility": float(utility.mean()),
                    "oracle_subset_overlap": float(overlap.mean()),
                    "oracle_utility": float(oracle_prediction_utility.mean()),
                }
            )
    choices.update(
        {
            "predicted_scores": predicted_scores.numpy(),
            "predicted_importance_logits": importance_logits.numpy(),
            "predicted_alpha": predicted_alpha.numpy(),
            "dataset_index": data["dataset_index"],
            "paths": data["paths"],
        }
    )
    return pd.DataFrame(rows), choices


def static_masks_for_train(
    train: dict[str, np.ndarray], keep_counts: list[int]
) -> dict[int, torch.Tensor]:
    score = torch.from_numpy(
        np.nan_to_num(train["scores"], nan=-1.0).clip(-1, 1).mean(axis=0)
    )[None]
    drop = torch.from_numpy(train["drops"].mean(axis=0))[None]
    return {
        keep: selected_masks(score, drop, subset_masks(score.shape[-1], keep))[0][0]
        for keep in keep_counts
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    keep_counts = [int(value) for value in args.keep_counts.split(",")]
    train = load_router_data(args.train_dir, args.layer_index)
    validation = load_router_data(args.val_dir, args.layer_index)
    test = load_router_data(args.test_dir, args.layer_index)
    joint_train = (
        load_joint_teacher(args.joint_train_dir)
        if args.joint_train_dir is not None
        else None
    )
    joint_validation = (
        load_joint_teacher(args.joint_val_dir)
        if args.joint_val_dir is not None
        else None
    )
    if joint_train is not None:
        assert_joint_alignment(train, joint_train)
    if joint_validation is not None:
        assert_joint_alignment(validation, joint_validation)
    heads = train["scores"].shape[-1]
    if any(keep < 1 or keep >= heads for keep in keep_counts):
        raise ValueError(f"keep counts must be in [1,{heads - 1}]")
    mean = train["global"].mean(axis=0)
    std = train["global"].std(axis=0).clip(min=1e-5)
    feature, scores, drops, alpha = tensors(train, mean, std)
    training_tensors = [feature, scores, drops, alpha]
    if joint_train is not None:
        training_tensors.extend(
            [
                torch.from_numpy(joint_train["candidate_masks"].astype(bool)),
                torch.from_numpy(joint_train["joint_kl"].astype(np.float32)),
            ]
        )
    loader = DataLoader(
        TensorDataset(*training_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    device = torch.device(args.device)
    masks_by_budget = {
        keep: subset_masks(heads, keep, device=device) for keep in keep_counts
    }
    model = DynamicCFSRouter(
        global_dim=feature.shape[-1],
        heads=heads,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    static_masks = static_masks_for_train(train, keep_counts)
    history, best_state, best_utility = [], None, -np.inf
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = []
        for batch in loader:
            global_feature, target_scores, batch_drops, target_alpha = batch[:4]
            joint_masks = batch[4].to(device) if joint_train is not None else None
            joint_kl = batch[5].to(device) if joint_train is not None else None
            global_feature = global_feature.to(device)
            target_scores = target_scores.to(device)
            batch_drops = batch_drops.to(device)
            target_alpha = target_alpha.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_scores, importance_logits, predicted_alpha = model(global_feature)
            terms = loss_terms(
                predicted_scores,
                importance_logits,
                predicted_alpha,
                target_scores,
                batch_drops,
                target_alpha,
                masks_by_budget,
                args.target_temperature,
                args.subset_temperature,
                joint_masks,
                joint_kl,
                args.joint_temperature,
            )
            loss = (
                terms["listwise"]
                + args.score_weight * terms["score"]
                + args.importance_weight * terms["importance"]
                + args.alpha_weight * terms["alpha"]
                + args.structured_weight * terms["structured"]
                + args.joint_weight * terms.get("joint", 0.0)
            )
            loss.backward()
            optimizer.step()
            epoch_rows.append({key: float(value.detach()) for key, value in terms.items()})
            epoch_rows[-1]["loss"] = float(loss.detach())
        validation_frame, _ = evaluate(
            model, validation, mean, std, keep_counts, static_masks, device
        )
        router_rows = validation_frame[
            validation_frame.policy == "dynamic_cfs_router"
        ]
        utility = float(router_rows.pairwise_surrogate_utility.mean())
        joint_regret = (
            joint_validation_regret(
                model, validation, joint_validation, mean, std, device
            )
            if joint_validation is not None
            else None
        )
        selection_score = -joint_regret if joint_regret is not None else utility
        row = {
            "epoch": epoch,
            "val_router_utility": utility,
            "val_joint_regret": joint_regret,
        }
        for key in epoch_rows[0]:
            row[f"train_{key}"] = float(np.mean([item[key] for item in epoch_rows]))
        history.append(row)
        if selection_score > best_utility:
            best_utility = selection_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
                f"val_utility={utility:.4f}"
                + (
                    f" val_joint_regret={joint_regret:.6f}"
                    if joint_regret is not None
                    else ""
                )
            )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_frame, _ = evaluate(
        model, validation, mean, std, keep_counts, static_masks, device
    )
    test_frame, predictions = evaluate(
        model, test, mean, std, keep_counts, static_masks, device
    )
    validation_frame.to_csv(args.output_dir / "validation_budget_summary.csv", index=False)
    test_frame.to_csv(args.output_dir / "test_budget_summary.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    np.savez_compressed(args.output_dir / "test_router_predictions.npz", **predictions)
    torch.save(
        {
            "state_dict": best_state,
            "global_mean": mean,
            "global_std": std,
            "heads": heads,
            "global_dim": feature.shape[-1],
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "static_masks": {key: value for key, value in static_masks.items()},
            "args": vars(args),
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "teacher": "pairwise CFS graph",
                "runtime_input": "pre-attention residual CLS+patch mean+patch variance",
                "post_head_leakage": False,
                "heads": heads,
                "keep_counts": keep_counts,
                "train_images": len(train["global"]),
                "validation_images": len(validation["global"]),
                "test_images": len(test["global"]),
                "best_validation_utility": best_utility,
                "joint_subset_supervision": joint_train is not None,
                "evaluation_warning": (
                    "pairwise_surrogate_utility uses oracle pair outcomes for the "
                    "selected set; real joint forward intervention is evaluated next"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(test_frame.to_string(index=False))
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
