"""Compare the Functional Basis checkpoint with the matched CE CFS audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_cfs_developmental_oracles import analyze_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce-dir", type=Path, required=True)
    parser.add_argument("--functional-basis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = (3, 7, 11)
    rows = []
    for policy, directory in (
        ("ce_baseline", args.ce_dir),
        ("functional_basis", args.functional_basis_dir),
    ):
        for position, layer in enumerate(layers):
            rows.append(
                analyze_layer(
                    policy, directory, position, layer, args.minimum_drop_kl
                )
            )

    by_layer = pd.DataFrame(rows)
    macro = by_layer.groupby("policy", as_index=False).agg(
        {
            "valid_source_fraction": "mean",
            "mean_drop_kl": "mean",
            "mean_best_substitution_kl": "mean",
            "raw_oracle": "mean",
            "matched_best_of_2": "mean",
            "source_fraction_s_gt_025": "mean",
            "representative_concentration": "mean",
            "representative_entropy": "mean",
            "slice_accuracy": "first",
        }
    )
    by_layer.to_csv(args.output_dir / "functional_basis_oracle_by_layer.csv", index=False)
    macro.to_csv(args.output_dir / "functional_basis_oracle_macro.csv", index=False)
    print("BY_LAYER")
    print(by_layer.to_string(index=False))
    print("MACRO")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
