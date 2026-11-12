"""Layer-wise and image-bootstrap diagnostics for Phase-A predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_representative_predictor import (
    flatten_examples,
    load_shards,
    select_layer,
    static_representatives,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--train-layer-index", type=int)
    parser.add_argument("--val-layer-index", type=int)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def selections(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> dict[str, np.ndarray | None]:
    heads = int(validation["heads_count"])
    candidates = np.arange(heads)[None, :]
    valid = candidates != validation["source"][:, None]
    static = static_representatives(
        train,
        int(train["layers_count"]),
        heads,
    )
    return {
        "random_expected": None,
        "current_l2": np.where(valid, validation["l2"], np.inf).argmin(axis=1),
        "highest_taylor_importance": np.where(
            valid, validation["taylor"], -np.inf
        ).argmax(axis=1),
        "static_train_representative": static[
            validation["layer"], validation["source"]
        ],
        "prehead_predictor": prediction,
        "functional_oracle": validation["label"],
    }


def realized(
    scores: np.ndarray,
    representative: np.ndarray | None,
) -> np.ndarray:
    if representative is None:
        return np.nanmean(scores, axis=1)
    return scores[np.arange(len(scores)), representative]


def bootstrap_difference(
    sample: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    unique = np.unique(sample)
    per_image_left = np.asarray([left[sample == item].mean() for item in unique])
    per_image_right = np.asarray([right[sample == item].mean() for item in unique])
    difference = per_image_left - per_image_right
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        difference,
        size=(repeats, len(difference)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = flatten_examples(
        select_layer(load_shards(args.train_dir), args.train_layer_index),
        args.minimum_drop_kl,
    )
    validation = flatten_examples(
        select_layer(load_shards(args.val_dir), args.val_layer_index),
        args.minimum_drop_kl,
    )
    archive = np.load(args.predictor_dir / "predictions.npz")
    prediction = archive["prediction"]
    policies = selections(train, validation, prediction)
    values = {
        name: realized(validation["scores"], representative)
        for name, representative in policies.items()
    }

    rows = []
    for layer in np.unique(validation["layer"]):
        selector = validation["layer"] == layer
        for name, value in values.items():
            rows.append(
                {
                    "layer": int(layer),
                    "policy": name,
                    "mean_recovery": float(value[selector].mean()),
                    "median_recovery": float(np.median(value[selector])),
                }
            )
    pd.DataFrame(rows).to_csv(args.output_dir / "layer_summary.csv", index=False)

    comparisons = []
    for baseline in (
        "static_train_representative",
        "highest_taylor_importance",
    ):
        result = bootstrap_difference(
            validation["sample"],
            values["prehead_predictor"],
            values[baseline],
            args.bootstrap,
            args.seed,
        )
        comparisons.append({"comparison": f"predictor-minus-{baseline}", **result})
    pd.DataFrame(comparisons).to_csv(
        args.output_dir / "bootstrap_comparisons.csv",
        index=False,
    )
    layer_comparisons = []
    for layer in np.unique(validation["layer"]):
        selector = validation["layer"] == layer
        for baseline in (
            "static_train_representative",
            "highest_taylor_importance",
        ):
            result = bootstrap_difference(
                validation["sample"][selector],
                values["prehead_predictor"][selector],
                values[baseline][selector],
                args.bootstrap,
                args.seed + int(layer),
            )
            layer_comparisons.append(
                {
                    "layer": int(layer),
                    "comparison": f"predictor-minus-{baseline}",
                    **result,
                }
            )
    pd.DataFrame(layer_comparisons).to_csv(
        args.output_dir / "layer_bootstrap_comparisons.csv",
        index=False,
    )
    print(pd.DataFrame(comparisons).to_string(index=False))
    pivot = pd.DataFrame(rows).pivot(
        index="layer",
        columns="policy",
        values="mean_recovery",
    )
    print(pivot.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
