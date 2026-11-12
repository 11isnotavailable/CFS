"""Selectively extract the official BREEDS Living-17 task from ImageNet-1K.

The 17 x 4 subclass indices are the deterministic ``split=None`` output of
MadryLab robustness.tools.breeds_helpers.make_living17.  We use all four fine
classes in each superclass, producing the same images with either 17 coarse or
68 fine labels.  The full ImageNet archive is streamed; it is never unpacked.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


LIVING17_CLASS_INDICES = (
    (27, 29, 26, 28), (37, 34, 33, 35), (41, 44, 47, 38),
    (60, 57, 65, 61), (76, 72, 74, 77), (81, 83, 82, 80),
    (88, 90, 87, 89), (118, 120, 119, 121), (163, 154, 257, 157),
    (272, 271, 270, 269), (280, 279, 277, 278), (282, 285, 283, 284),
    (297, 295, 296, 294), (305, 306, 302, 303), (325, 321, 324, 322),
    (368, 365, 366, 367), (377, 380, 381, 379),
)

SUPERCLASS_NAMES = (
    "salamander", "turtle", "lizard", "snake", "spider", "grouse",
    "parrot", "crab", "dog", "wolf", "fox", "domestic cat", "bear",
    "beetle", "butterfly", "ape", "monkey",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tar", type=Path, required=True)
    parser.add_argument("--val-tar", type=Path, required=True)
    parser.add_argument("--val-ground-truth", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--class-info", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def safe_write(source, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            output.write(block)


def scalar(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def extract_train(archive_path: Path, output_root: Path, selected: set[str]) -> None:
    complete = {
        wnid for wnid in selected
        if (output_root / wnid / ".complete").exists()
    }
    if complete == selected:
        print("train extraction already complete", flush=True)
        return
    with tarfile.open(archive_path, "r:") as outer:
        for member in outer:
            wnid = Path(member.name).stem
            if wnid not in selected or wnid in complete or not member.isfile():
                continue
            stream = outer.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Could not read nested archive {member.name}")
            target = output_root / wnid
            target.mkdir(parents=True, exist_ok=True)
            count = 0
            with tarfile.open(fileobj=stream, mode="r|") as inner:
                for image in inner:
                    if not image.isfile():
                        continue
                    image_stream = inner.extractfile(image)
                    if image_stream is None:
                        continue
                    safe_write(image_stream, target / Path(image.name).name)
                    count += 1
            (target / ".complete").write_text(str(count), encoding="utf-8")
            complete.add(wnid)
            print(f"train {len(complete)}/{len(selected)} {wnid} images={count}", flush=True)
    missing = selected - complete
    if missing:
        raise RuntimeError(f"Missing train class archives: {sorted(missing)}")


def extract_val(
    archive_path: Path,
    output_root: Path,
    ground_truth: list[int],
    id_to_wnid: dict[int, str],
    selected: set[str],
) -> None:
    marker = output_root / ".complete"
    if marker.exists():
        print("validation extraction already complete", flush=True)
        return
    pattern = re.compile(r"(\d{8})\.JPEG$")
    counts = {wnid: 0 for wnid in selected}
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = pattern.search(member.name)
            if not match:
                continue
            image_index = int(match.group(1)) - 1
            class_id = ground_truth[image_index]
            wnid = id_to_wnid[class_id]
            if wnid not in selected:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            safe_write(stream, output_root / wnid / Path(member.name).name)
            counts[wnid] += 1
    if any(value != 50 for value in counts.values()):
        raise RuntimeError(f"Expected 50 validation images per class, got {counts}")
    marker.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"validation classes={len(counts)} images={sum(counts.values())}", flush=True)


def main() -> None:
    args = parse_args()
    class_rows = json.loads(args.class_info.read_text(encoding="utf-8"))
    by_index = {int(index): (str(wnid), str(words)) for index, wnid, words in class_rows}
    groups = []
    fine_order = []
    for coarse_index, indices in enumerate(LIVING17_CLASS_INDICES):
        members = []
        for class_index in indices:
            wnid, words = by_index[class_index]
            members.append({"imagenet_index": class_index, "wnid": wnid, "words": words})
            fine_order.append(wnid)
        groups.append({
            "coarse_index": coarse_index,
            "name": SUPERCLASS_NAMES[coarse_index],
            "members": members,
        })
    selected = set(fine_order)
    fine_mapping = {wnid: index for index, wnid in enumerate(fine_order)}
    coarse_mapping = {
        member["wnid"]: group["coarse_index"]
        for group in groups for member in group["members"]
    }

    train_root = args.output_root / "train"
    val_root = args.output_root / "val"
    extract_train(args.train_tar, train_root, selected)
    ground_truth = [
        int(line) for line in args.val_ground_truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_synsets = loadmat(
        args.meta, squeeze_me=True, struct_as_record=False
    )["synsets"]
    id_to_wnid = {
        int(scalar(item.ILSVRC2012_ID)): str(scalar(item.WNID))
        for item in np.ravel(raw_synsets)
    }
    extract_val(args.val_tar, val_root, ground_truth, id_to_wnid, selected)

    payload = {
        "benchmark": "BREEDS Living-17 without source/target subclass split",
        "source": "MadryLab robustness make_living17(info_dir, split=None)",
        "train_root": str(train_root),
        "val_root": str(val_root),
        "fine_classes": 68,
        "coarse_classes": 17,
        "fine_class_to_idx": fine_mapping,
        "coarse_class_to_idx": coarse_mapping,
        "groups": groups,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"metadata={args.metadata}", flush=True)


if __name__ == "__main__":
    main()
