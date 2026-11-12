"""Extract the ILSVRC-2012 validation archive into ImageFolder layout.

The public validation tar stores all images in one directory.  This utility
uses the official devkit ground truth and synset metadata to place each image
under its WordNet ID, yielding the class layout expected by torchvision's
ImageFolder and the existing CFS collector.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-tar", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def scalar(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def safe_write(source, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while block := source.read(1024 * 1024):
            output.write(block)


def main() -> None:
    args = parse_args()
    marker = args.output_root / ".complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("images") != 50_000 or payload.get("classes") != 1_000:
            raise RuntimeError(f"Invalid completion marker: {payload}")
        print(f"validation extraction already complete: {args.output_root}")
        return

    ground_truth = [
        int(line)
        for line in args.ground_truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ground_truth) != 50_000:
        raise RuntimeError(f"Expected 50000 labels, got {len(ground_truth)}")

    raw_synsets = loadmat(
        args.meta, squeeze_me=True, struct_as_record=False
    )["synsets"]
    id_to_wnid = {
        int(scalar(item.ILSVRC2012_ID)): str(scalar(item.WNID))
        for item in np.ravel(raw_synsets)
        if int(scalar(item.num_children)) == 0
    }
    if len(id_to_wnid) != 1_000:
        raise RuntimeError(f"Expected 1000 leaf synsets, got {len(id_to_wnid)}")

    pattern = re.compile(r"(\d{8})\.JPEG$")
    counts = {wnid: 0 for wnid in id_to_wnid.values()}
    written = 0
    with tarfile.open(args.val_tar, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = pattern.search(member.name)
            if match is None:
                continue
            image_index = int(match.group(1)) - 1
            wnid = id_to_wnid[ground_truth[image_index]]
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member.name}")
            safe_write(source, args.output_root / wnid / Path(member.name).name)
            counts[wnid] += 1
            written += 1

    if written != 50_000 or any(count != 50 for count in counts.values()):
        raise RuntimeError(
            f"Invalid extraction: images={written}, "
            f"class count range=({min(counts.values())},{max(counts.values())})"
        )
    marker.write_text(
        json.dumps({"images": written, "classes": len(counts)}, indent=2),
        encoding="utf-8",
    )
    print(f"extracted images={written} classes={len(counts)} to {args.output_root}")


if __name__ == "__main__":
    main()
