#!/usr/bin/env bash
set -euo pipefail
cd "${1:?experiment root}"
PYTHON="${CFS_PYTHON:-/data/cfs_rebuttal_20261114/venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CFS_GPU:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
mkdir -p runs
for kind in gating joint_only_r2; do
  for seed in 17 23 41; do
    "$PYTHON" scripts/train_learned_gating.py --data learned_gating_inputs_20261115 --output "runs/${kind}_seed${seed}" --kind "$kind" --seed "$seed" > "runs/${kind}_seed${seed}.log" 2>&1
  done
done
date -u > TRAINING_COMPLETE
