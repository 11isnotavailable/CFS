from scripts.prepare_breeds_living17 import LIVING17_CLASS_INDICES
from scripts.analyze_task_demand_interaction import bootstrap_interaction
from scripts.train_vit_task_demand import stratified_indices
import pandas as pd


def test_living17_is_balanced_and_unique():
    assert len(LIVING17_CLASS_INDICES) == 17
    assert {len(group) for group in LIVING17_CLASS_INDICES} == {4}
    flattened = [item for group in LIVING17_CLASS_INDICES for item in group]
    assert len(flattened) == 68
    assert len(set(flattened)) == 68


def test_fine_stratified_split_is_deterministic_and_complete():
    targets = [0] * 10 + [1] * 20 + [2] * 30
    first_train, first_holdout = stratified_indices(targets, 0.1, 17)
    second_train, second_holdout = stratified_indices(targets, 0.1, 17)
    assert first_train == second_train
    assert first_holdout == second_holdout
    assert len(first_holdout) == 6
    assert sorted(first_train + first_holdout) == list(range(len(targets)))


def test_bootstrap_interaction_uses_difference_in_differences():
    rows = []
    values = {
        ("coarse", "base"): [1.0, 2.0],
        ("coarse", "large"): [3.0, 4.0],
        ("fine", "base"): [2.0, 4.0],
        ("fine", "large"): [3.0, 5.0],
    }
    for (task, size), observations in values.items():
        for sample, value in enumerate(observations):
            rows.append({"task": task, "model_size": size, "sample": sample, "x": value})
    result = bootstrap_interaction(pd.DataFrame(rows), "x", repeats=100, seed=17)
    assert result["interaction"] == 1.0
