"""Inspect the public ImageNet-100 archive against the ILSVRC hierarchy."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scalar(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def main() -> None:
    args = parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        counts: dict[str, int] = {}
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) == 3 and not name.endswith("/"):
                counts[parts[1]] = counts.get(parts[1], 0) + 1

    raw = loadmat(args.meta, squeeze_me=True, struct_as_record=False)["synsets"]
    records = {}
    field_names = sorted(
        name for name in dir(np.ravel(raw)[0]) if not name.startswith("_")
    )
    for item in np.ravel(raw):
        wnid = str(scalar(item.WNID))
        children = np.atleast_1d(item.children).astype(int).tolist()
        records[wnid] = {
            "ilsvrc_id": int(scalar(item.ILSVRC2012_ID)),
            "words": str(scalar(item.words)),
            "num_children": int(scalar(item.num_children)),
            "children": children,
            "wordnet_height": int(scalar(item.wordnet_height)),
        }

    missing = sorted(set(counts) - set(records))
    selected = [
        {"wnid": wnid, "images": counts[wnid], **records[wnid]}
        for wnid in sorted(counts)
        if wnid in records
    ]
    payload = {
        "archive": str(args.archive),
        "meta": str(args.meta),
        "meta_fields": field_names,
        "classes": len(counts),
        "images": sum(counts.values()),
        "missing_wnids": missing,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("classes", "images", "missing_wnids")}))
    for row in selected:
        print(f"{row['wnid']}\t{row['images']}\t{row['words']}")


if __name__ == "__main__":
    main()
