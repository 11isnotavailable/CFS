"""Derive a deterministic semantic coarse-label partition from ILSVRC metadata."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import squareform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument(
        "--linkage", choices=("average", "complete", "single"), default="average"
    )
    parser.add_argument(
        "--balanced", action=argparse.BooleanOptionalAction, default=True,
        help="Constrain every semantic cluster to the same number of fine classes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scalar(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def load_hierarchy(path: Path):
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)["synsets"]
    records = {}
    id_to_wnid = {}
    for item in np.ravel(raw):
        node_id = int(scalar(item.ILSVRC2012_ID))
        wnid = str(scalar(item.WNID))
        children = tuple(int(value) for value in np.atleast_1d(item.children))
        records[node_id] = {
            "wnid": wnid,
            "words": str(scalar(item.words)),
            "children": children,
            "height": int(scalar(item.wordnet_height)),
        }
        id_to_wnid[node_id] = wnid
    return records, {wnid: node_id for node_id, wnid in id_to_wnid.items()}


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selected_wnids = tuple(row["wnid"] for row in inventory["selected"])
    selected = set(selected_wnids)
    records, wnid_to_id = load_hierarchy(args.meta)

    @lru_cache(maxsize=None)
    def selected_descendants(node_id: int) -> frozenset[str]:
        record = records[node_id]
        values = {record["wnid"]} & selected
        for child in record["children"]:
            if child in records:
                values.update(selected_descendants(child))
        return frozenset(values)

    roots = sorted(
        (
            (len(selected_descendants(node_id)), records[node_id]["height"], node_id)
            for node_id in records
            if len(selected_descendants(node_id)) == len(selected)
        ),
        key=lambda item: (item[1], item[2]),
    )
    if not roots:
        raise RuntimeError("No hierarchy node covers every selected ImageNet-100 class")
    root = roots[0][2]

    # The WordNet root can have a high branching factor, so a literal level cut
    # is usually extremely imbalanced. Cluster the selected leaves using only
    # shortest-path distances in the fixed ILSVRC hierarchy instead. This is
    # deterministic, independent of images/model outputs, and gives exactly K
    # semantic-granularity groups.
    node_ids = sorted(records)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    rows, columns = [], []
    for parent, record in records.items():
        for child in record["children"]:
            if child in node_index:
                rows.extend((node_index[parent], node_index[child]))
                columns.extend((node_index[child], node_index[parent]))
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(len(node_ids), len(node_ids)),
    )
    leaf_ids = [wnid_to_id[wnid] for wnid in selected_wnids]
    leaf_indices = [node_index[node_id] for node_id in leaf_ids]
    distances = shortest_path(graph, directed=False, indices=leaf_indices)
    pairwise = distances[:, leaf_indices]
    if not np.isfinite(pairwise).all():
        raise RuntimeError("Disconnected nodes in ILSVRC hierarchy")
    if args.balanced:
        if len(selected_wnids) % args.groups:
            raise ValueError("Balanced grouping requires classes divisible by groups")
        capacity = len(selected_wnids) // args.groups
        medoids = [0]
        while len(medoids) < args.groups:
            nearest = pairwise[:, medoids].min(axis=1)
            nearest[medoids] = -1
            medoids.append(int(np.argmax(nearest)))
        labels = np.zeros(len(selected_wnids), dtype=int)
        for _ in range(100):
            slots = np.repeat(np.asarray(medoids, dtype=int), capacity)
            row_indices, slot_indices = linear_sum_assignment(pairwise[:, slots])
            updated_labels = np.empty_like(labels)
            updated_labels[row_indices] = slot_indices // capacity
            updated_medoids = []
            for label in range(args.groups):
                members = np.flatnonzero(updated_labels == label)
                within = pairwise[np.ix_(members, members)].sum(axis=1)
                updated_medoids.append(int(members[np.argmin(within)]))
            labels = updated_labels
            if updated_medoids == medoids:
                break
            medoids = updated_medoids
    else:
        hierarchy = linkage(
            squareform(pairwise, checks=False), method=args.linkage
        )
        labels = cut_tree(hierarchy, n_clusters=[args.groups]).reshape(-1)
    raw_groups = []
    for label in sorted(set(labels.tolist())):
        members = frozenset(
            selected_wnids[index]
            for index in np.flatnonzero(labels == label)
        )
        ancestors = [
            node_id for node_id in records
            if members.issubset(selected_descendants(node_id))
        ]
        representative = min(
            ancestors,
            key=lambda node_id: (
                len(selected_descendants(node_id)),
                records[node_id]["height"],
                node_id,
            ),
        )
        raw_groups.append({"node": representative, "members": members})
    groups = sorted(raw_groups, key=lambda group: min(group["members"]))
    mapping = {}
    summary = []
    for label, group in enumerate(groups):
        record = records[group["node"]]
        members = sorted(group["members"])
        for wnid in members:
            if wnid in mapping:
                raise RuntimeError(f"Duplicate assignment for {wnid}")
            mapping[wnid] = label
        summary.append({
            "label": label,
            "ancestor_wnid": record["wnid"],
            "ancestor_words": record["words"],
            "classes": len(members),
            "members": members,
        })
    if set(mapping) != selected:
        raise RuntimeError("Coarse partition does not cover the selected classes")

    payload = {
        "method": (
            "balanced k-medoids over ILSVRC2012 WordNet shortest-path distance"
            if args.balanced else
            f"{args.linkage}-linkage clustering of ILSVRC2012 WordNet shortest-path distance"
        ),
        "requested_groups": args.groups,
        "realized_groups": len(groups),
        "root": {
            "wnid": records[root]["wnid"],
            "words": records[root]["words"],
        },
        "class_to_idx": mapping,
        "groups": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"requested": args.groups, "realized": len(groups)}))
    for group in summary:
        print(
            f"{group['label']:02d}\t{group['classes']:3d}\t"
            f"{group['ancestor_wnid']}\t{group['ancestor_words']}"
        )
        print("    " + "; ".join(
            records[wnid_to_id[wnid]]["words"].split(",")[0]
            for wnid in group["members"]
        ))


if __name__ == "__main__":
    main()
