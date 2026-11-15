# GPT-2 cross-model routing protocol

Public `openai-community/gpt2`, exact downloaded revision/hashes in provenance.
Never train the backbone. Primary layer11, optional layer8 only after timing
and numerical checks. Same exact R2 architecture/utility and original loss_terms;
no A3, residual, adapter or ranking-loss search. Seeds17/23/41. Main hard gating.

Reuse FineWeb-Edu sample-10BT parquet from the previous language experiments.
Freeze 2,432 unique documents/prefixes: train2048/tune128/test256, seed20260802.
One 128-token prefix/document, position127 predicts token128. This fixed-position
adaptation differs from controlled GPT-2 scaling's random-eight position mean;
it is not a reproduction of that scaling table. No padding, no KV reuse; reject
padded inputs. Only tokens0..127 enter features/model. Document and prefix hashes
are disjoint. Report selected-position NLL/PPL, not whole-corpus perplexity.

Feature adaptation: [current; prefix mean; population variance] of ln_1 output
at the routed block, including current token. This matches ViT's norm1 capture
site but replaces CLS/patch summaries. Float32 features, train-only mean/std.
All learned methods use identical features. Primary backbone float32, TF32 off,
temperature1 reference-to-intervention KL, FP32 log-softmax, clamp_min0 exactly
as existing GPT collection. Cache c_proj input and block residual, preserve
projection bias once; rerun only necessary suffix. Validate against full forward.

Pair oracle: alpha0..3,16 points train/31 tune (ViT routing convention; the
controlled GPT collector used16 throughout). Save raw drops/substitution/alpha.
CFS=1-Dreplacement/max(Ddrop,1e-12), train nan->-1 clip[-1,1], original utility
clip[0,1]. Drop>1e-5 is a reported validity diagnostic, not a new training filter.
Alpha=1 with dropped source equals deletion-only; alpha=0 removes both branches.

Teacher: original seven anchor/local-swap candidate_pool,48 masks/K3/6/9.
First fit a seed17 pair-only bootstrap R2 to generate the old-router anchor;
use training-derived static CFS graph. This anchor model is not a main result.
Pilot uses train256/tune64 and20epochs; full uses train2048/tune128 and60epochs.
Final models train from scratch, not from pilot or anchor. Cache pair data and
joint teachers resumably; freeze candidate provenance per stage.

Original R2 loss: listwise+.5score+.5importance+.1alpha+structured+.5joint,
exact existing functions/temperatures. Gating and A0: .5joint; A0 alpha frozen.
AdamW3e-4,decay1e-4,batch64, fixed epochs. Gating822460 params vs R2825091;
A0 total825091, optimized824962. Report this small mismatch (~0.3%).

Main checkpoint selection: real hard routing validation KL using exact full
220/924/220 subset search, evaluated every10epochs (pilot10/20, full10..60),
equal mean across budgets, earlier epoch on ties. This explicitly improves the
old48-candidate validation proxy without changing the R2 objective/architecture.
Bootstrap anchor selection uses oracle pairwise utility at the same checkpoints.
All test decisions occur only after nine model selections for that layer lock.
No test-based method, budget weight, epoch, compensation or layer choice.

Taylor: original |gate*dCE/dgate|, same target-token NLL, absolute AFTER summing
activation-gradient products. It uses a supervised backward and the true next
token, so label it an analytical privileged baseline, not deployable forward-only
generation. R2/Gating/A0 do not see that target at routing inference.

Stages:32-prefix smoke for correctness/cost; pilot256/64; full only after stable
finite pilot and estimated collection fits a modest GPU budget. Layer8 repeats
all teachers/training with identical data if measured cost permits; otherwise
document why deferred. Freeze sample count before test. Save all logs/weights.

Evaluate three seeds, all budgets, same dense reference. Mean KL,NLL,exp(mean
token NLL),paired document bootstrap2000 seed42 (one token/document),per-seed
tables,latency/parameters. Do not average per-token exp(NLL) as perplexity.
No sparse-kernel speedup claim. Cross-task absolute KL is not directly comparable.
