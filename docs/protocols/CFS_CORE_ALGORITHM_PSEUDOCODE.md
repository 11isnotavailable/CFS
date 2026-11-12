# CFS Core Algorithm and Pseudocode

> The executable implementation is available in `src/` and `scripts/`; compact
> paper-result tables are available in `results/`.

## 1. Functional intervention

For layer \(l\), write the post-output-projection contribution of head \(i\) as

\[
c_{l,i}(x)=A_{l,i}(x)V_{l,i}(x)W^{(i)}_{O,l}.
\]

With gates \(g\in\mathbb R^H\), the downstream network is

\[
f_l(x;g)=G_l\!\left(r_l(x)+\sum_i g_i c_{l,i}(x)\right).
\]

For temperature \(T\), the intervention error is always measured in the
reference-to-intervention direction:

\[
D_l(x;g)=T^2D_{\mathrm{KL}}\!\left(
\operatorname{softmax}(f_l(x;\mathbf 1)/T)
\;\|\;
\operatorname{softmax}(f_l(x;g)/T)
\right).
\]

The main vision experiments use \(T=1\).

## 2. Algorithm 1: per-input CFS matrix

```text
Algorithm CFS-MATRIX(model f, input x, layer l, alpha grid A,
                     valid threshold delta=1e-5, temperature T=1)

  dense_logits <- FORWARD(f, x, gates=ones(H), intervene_at=l)
  p <- softmax(dense_logits / T)

  initialize drop_kl[H]
  initialize replace_kl[H,H] <- NaN
  initialize alpha_star[H,H] <- NaN
  initialize S[H,H] <- NaN

  for source i in {1,...,H}:
      g <- ones(H); g[i] <- 0
      q <- softmax(FORWARD(f, x, g, l) / T)
      drop_kl[i] <- T^2 * KL(p || q)

      if drop_kl[i] <= delta:
          continue  # invalid source for normalized summaries

      for candidate j in {1,...,H} excluding i:
          best_error <- +infinity
          best_alpha <- NaN

          for alpha in A:
              g <- ones(H)
              g[i] <- 0
              g[j] <- alpha
              q <- softmax(FORWARD(f, x, g, l) / T)
              error <- T^2 * KL(p || q)

              if error < best_error:
                  best_error <- error
                  best_alpha <- alpha

          replace_kl[i,j] <- best_error
          alpha_star[i,j] <- best_alpha
          S[i,j] <- 1 - best_error / max(drop_kl[i], epsilon)

  valid <- (drop_kl > delta)
  return S, drop_kl, replace_kl, alpha_star, valid
```

Interpretation: \(S=1\) is perfect recovery, \(S=0\) is no better than direct
deletion, and \(S<0\) is worse than deletion. CFS is directed and need not be
transitive.

## 3. Algorithm 2: canonical summary metrics

```text
Algorithm SUMMARIZE-CFS({S_x, drop_x, replace_x, valid_x}, k, tau)

  for every input x and valid source i:
      candidate_scores <- {S_x[i,j] : j != i}
      raw_source[x,i] <- max(candidate_scores)
      best_kl[x,i] <- min_j replace_x[i,j]
      matched_k[x,i] <- exact expected maximum of a uniformly sampled
                        size-k candidate subset

  RawOracle <- mean(raw_source over valid sources)
  MatchedK <- mean(matched_k over valid sources)
  MeanDropKL <- mean(drop_x over valid sources)
  MeanBestSubKL <- mean(best_kl over valid sources)

  for every input x:
      WeightedRecovery[x] <-
          sum_i(drop_x[i] - best_kl[x,i]) / sum_i drop_x[i]

      R_x <- exact minimum retained set such that each valid source i
             is either retained or has some retained j with S_x[i,j] > tau
      Cover[x] <- |R_x|

      F_x[i,j] <- clip(S_x[i,j], 0, 1) for valid i and i != j
      F_x[i,i] <- 1 for valid i; invalid source rows <- 0
      sigma <- singular_values(F_x)
      p <- sigma / sum(sigma)
      EffectiveRank[x] <- exp(-sum_{r:p[r]>0} p[r] log p[r])

  CoverPerHead <- mean_x(Cover[x]) / H
  RankPerHead <- mean_x(EffectiveRank[x]) / H

  return RawOracle, MatchedK, MeanDropKL, MeanBestSubKL,
         mean(WeightedRecovery), CoverPerHead, RankPerHead
```

Matched-\(k\) is evaluated exactly from ordered scores rather than by Monte
Carlo sampling. Raw Oracle must be accompanied by Matched-\(k\) when candidate
head counts differ. A per-input cover is integer; its dataset mean need not be.

## 4. Algorithm 3: CFS teacher construction

```text
Algorithm BUILD-CFS-TEACHER(backbone, frozen examples X, layers L, alpha grid A)
  freeze backbone parameters and example order
  for x in X:
      for l in L:
          snapshot[x,l] <- CFS-MATRIX(backbone, x, l, A)
  persist raw per-input S, drop KL, replacement KL, alpha*, and validity mask
  persist split IDs, model/config hashes, intervention check, and protocol version
  return teacher snapshots
```

The all-ones gate must reproduce the dense model within numerical tolerance
before any teacher data are accepted.

## 5. Algorithm 4: final forward-only structured router

The final application uses the pre-attention residual only:

\[
z_l(x)=[X_{\mathrm{CLS}};operatorname{Mean}(X_{\mathrm{patch}});
\operatorname{Var}(X_{\mathrm{patch}})].
\]

It predicts pair scores \(\hat S_{i\rightarrow j}\), pair calibration
\(\hat\alpha_{ij}\), and normalized importance \(\hat w_i\). For retained set
\(R\),

\[
\hat U(R\mid x)=\sum_i\hat w_i(x)
\begin{cases}
1,&i\in R,\\
\max_{j\in R}\operatorname{clip}(\hat S_{i\rightarrow j},0,1),&i\notin R.
\end{cases}
\]

```text
Algorithm ROUTE(residual X_l, budget K, trained router theta)
  z <- concat(CLS(X_l), mean(patch(X_l)), variance(patch(X_l)))
  z <- normalize_with_frozen_train_statistics(z)
  S_hat, alpha_hat, importance_hat <- theta(z, head identities)

  for every subset R of {1,...,H} with |R|=K:
      utility[R] <- structured_utility(R, S_hat, importance_hat)

  R_star <- argmax_R utility[R]       # exact for H=12 and K in {3,6,9}
  gates <- indicator_vector(R_star)   # primary hard-routing result
  return gates, R_star, S_hat, alpha_hat
```

The strongest R2 router retains pairwise teacher losses and adds direct
joint-subset supervision from true simultaneous-intervention KL. This matters
because independently good pair substitutions need not compose through a
nonlinear downstream network.

## 6. Non-negotiable implementation rules

1. Compute normalized recovery per input and source before averaging.
2. Keep source and representative roles directed.
3. Exclude invalid sources only from normalized summaries; retain raw deletion
   information for diagnosis.
4. Report absolute deletion and replacement KL beside normalized recovery.
5. Freeze probe examples, splits, head indexing, layer indexing, alpha grid,
   and output-space KL definition within each comparison.
6. Do not describe gradient probes as deployable forward-only routers.
7. Do not equate masked dense execution with realized hardware acceleration.

