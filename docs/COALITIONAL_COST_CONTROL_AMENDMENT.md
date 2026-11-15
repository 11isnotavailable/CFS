# Cost-control clarification, before inspecting test metrics (2026-11-16)

The primary39 models were already locked and their test teachers/evaluation
started. No test metric has been inspected to choose this addition. The
amortized GPU-path wall-time curve is an estimate, not literal equal measured
wall time for each subsampled label set. To satisfy the independent
intervention-evaluation-budget comparison, add12 Joint2 fits: budgets
floor(2500/31), floor(5000/31), floor(10000/31), floor(20000/31), seeds17/23/41.

Pair labels each require31 grid points; Joint2 labels require961=31x31.
Thus candidate-intervention counts agree up to integer rounding (<1%); dense
capture and source-drop measurement are shared fixed caches, excluded from this
relation-label budget and reported separately. Forward point counts do not
imply equal wall time, and backward count is zero for both exact-grid teachers.

These are additional training-data-only fits with the SAME architecture,
optimizer,2000steps and validation selection. Keep the original39-model lock,
all raw results and all previously specified comparisons unchanged. A separate
selection_lock_evaluations.json records these12 models before evaluating their
predictions. This is a declared budget-accounting control, not test-driven
algorithm search. Do not describe all51 models as locked before test labels
were first generated.
