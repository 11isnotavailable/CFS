"""Build the controlled width/depth CFS delivery table."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing W576/W768/W1152/D9/D18 CFS summaries.",
    )
    return parser.parse_args()


def load(output_dir, name):
    with (output_dir / name / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["aggregation"] == "three_layer_mean")


def main():
    output_dir = parse_args().output_dir
    mapping = [
        ("Width", "W576", "W576"),
        ("Width", "W768", "W768"),
        ("Width", "W1152", "W1152"),
        ("Depth", "D9", "D9"),
        ("Depth", "D12", "W768"),
        ("Depth", "D18", "D18"),
    ]
    rows = []
    for axis, setting, source in mapping:
        value = load(output_dir, source)
        rows.append({
            "axis": axis, "setting": setting,
            "parameters": int(value["parameters"]),
            "validation_loss": float(value["training_validation_loss"]),
            "raw_oracle": float(value["raw_oracle"]),
            "matched_2": float(value["matched_2"]),
            "cover_per_head": float(value["cover_per_head"]),
            "rank_per_head": float(value["rank_per_head"]),
        })
    csv_path = output_dir / "controlled_scaling_cfs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    header = "| Axis | Setting | Params | Val. Loss | Raw | Match-2 | Cover/H | Rank/H |"
    md = [header, "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['axis']} | {r['setting']} | {r['parameters']:,} | {r['validation_loss']:.4f} | {r['raw_oracle']:.4f} | {r['matched_2']:.4f} | {r['cover_per_head']:.4f} | {r['rank_per_head']:.4f} |")
    (output_dir / "CONTROLLED_SCALING_CFS.md").write_text("\n".join(md) + "\n")
    lines = [
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Axis & Setting & Params & Val. Loss & Raw & Match-2 & Cover/H & Rank/H \\",
        r"\midrule",
    ]
    for index, r in enumerate(rows):
        if index in (3,): lines.append(r"\midrule")
        lines.append(f"{r['axis']} & {r['setting']} & {r['parameters']:,} & {r['validation_loss']:.4f} & {r['raw_oracle']:.4f} & {r['matched_2']:.4f} & {r['cover_per_head']:.4f} & {r['rank_per_head']:.4f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "controlled_scaling_cfs_table.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
