"""Summarize tune-only CFS developmental training screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("vit_small_*_seed17_e75.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload["history"]
        last = history[-1]
        tail = history[-5:]
        rows.append({
            "policy": last["aux_policy"],
            "best_tune_accuracy": payload["best_tune_accuracy"],
            "final_tune_accuracy": last["tune_accuracy"],
            "last5_tune_accuracy": sum(x["tune_accuracy"] for x in tail) / len(tail),
            "final_tune_loss": last["tune_loss"],
            "elapsed_minutes": last["elapsed_minutes"],
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
