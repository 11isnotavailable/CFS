"""Compare CFS topology after fixed-checkpoint ViT-S post-training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_scaling_cfs import analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries, thresholds, samples = [], [], []
    for policy in ("none", "head_drop", "random_substitution", "self_cfs"):
        directory = args.root / f"cfs_vit_small_{policy}_seed17_post15_final512"
        result = analyze(policy, directory, args.minimum_drop_kl)
        summary, policy_thresholds = result[:2]
        policy_samples = result[2] if len(result) > 2 else []
        summaries.append(summary)
        thresholds.extend(policy_thresholds)
        samples.extend(policy_samples)

    summary = pd.DataFrame(summaries)
    threshold = pd.DataFrame(thresholds)
    sample = pd.DataFrame(samples)
    summary.to_csv(args.output_dir / "posttrain_cfs_summary.csv", index=False)
    threshold.to_csv(args.output_dir / "posttrain_cfs_thresholds.csv", index=False)
    if not sample.empty:
        sample.to_csv(args.output_dir / "posttrain_cfs_samples.csv", index=False)
    print(summary.to_string(index=False))
    print(threshold.to_string(index=False))


if __name__ == "__main__":
    main()
