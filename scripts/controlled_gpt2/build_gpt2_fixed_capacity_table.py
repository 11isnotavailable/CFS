"""Build the fixed-capacity head-decomposition CFS delivery table."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing H6/W768/H24 CFS summaries.",
    )
    return parser.parse_args()


def load(output_dir, name):
    with (output_dir / name / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["aggregation"] == "three_layer_mean")


def main():
    output_dir = parse_args().output_dir
    mapping = [("6x128", "H6"), ("12x64", "W768"), ("24x32", "H24")]
    rows = []
    for setting, source in mapping:
        value = load(output_dir, source)
        rows.append({
            "heads": setting, "parameters": int(value["parameters"]),
            "validation_loss": float(value["training_validation_loss"]),
            "raw_oracle": float(value["raw_oracle"]),
            "matched_2": float(value["matched_2"]),
            "cover_per_head": float(value["cover_per_head"]),
            "rank_per_head": float(value["rank_per_head"]),
            "effective_rank": float(value["spectral_rank"]),
        })
    with (output_dir / "fixed_capacity_cfs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    md = [
        "| Heads | Params | Val. Loss | Raw | Match-2 | Cover/H | Rank/H | Eff. Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['heads']} | {r['parameters']:,} | {r['validation_loss']:.4f} | {r['raw_oracle']:.4f} | {r['matched_2']:.4f} | {r['cover_per_head']:.4f} | {r['rank_per_head']:.4f} | {r['effective_rank']:.3f} |")
    (output_dir / "FIXED_CAPACITY_CFS.md").write_text("\n".join(md) + "\n")
    lines = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Heads & Params & Val. Loss & Raw & Match-2 & Cover/H & Rank/H & Eff. Rank \\",
        r"\midrule",
    ]
    for r in rows:
        label = "$" + r["heads"].replace("x", r"\times") + "$"
        lines.append(f"{label} & {r['parameters']:,} & {r['validation_loss']:.4f} & {r['raw_oracle']:.4f} & {r['matched_2']:.4f} & {r['cover_per_head']:.4f} & {r['rank_per_head']:.4f} & {r['effective_rank']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "fixed_capacity_cfs_table.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
