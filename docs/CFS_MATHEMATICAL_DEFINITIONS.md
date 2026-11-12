# CFS mathematical definitions and measurement protocol

## 1. Attention branch and downstream function

For input (x), layer (l), and attention head (i\in\{1,\ldots,H\}), define
the post-output-projection contribution

\[
c_{l,i}(x)=A_{l,i}(x)V_{l,i}(x)W^{(i)}_{O,l}.
\]

The multi-head attention output is additive:

\[
\operatorname{MHA}_l(x)=\sum_{i=1}^{H}c_{l,i}(x).
\]

Let (r_l(x)) denote everything entering the corresponding residual addition,
and let (G_l) denote all downstream computation from that addition to the
final logits. Introducing head gates (g\in\mathbb R^H),

\[
z_l(x;g)=r_l(x)+\sum_{i=1}^{H}g_i c_{l,i}(x),
\qquad
f_l(x;g)=G_l(z_l(x;g)).
\]

The unmodified network uses \(\mathbf 1=(1,\ldots,1)\). This formulation makes
CFS a property of the branch contribution *under downstream computation*, not
merely a similarity between current representations.

## 2. Output-space intervention error

At temperature (T), define

\[
p_x^{(T)}=\operatorname{softmax}\!\left(f_l(x;\mathbf1)/T\right),
\qquad
q_{x,g}^{(T)}=\operatorname{softmax}\!\left(f_l(x;g)/T\right).
\]

The intervention error is

\[
D_l(x;g)
=T^2D_{\mathrm{KL}}\!\left(p_x^{(T)}\,\|\,q_{x,g}^{(T)}\right).
\]

The implemented direction is always reference-to-intervention KL. Current ViT
experiments use (T=1).

## 3. Single-head deletion

Let (g^{\setminus i}) equal one except (g_i=0). The deletion error is

\[
D^{\mathrm{drop}}_{l,i}(x)=D_l(x;g^{\setminus i}).
\]

This measures the marginal effect of source head (i) on the final output; it
is not itself a substitutability score.

To prevent nearly inactive sources from creating unstable normalized ratios,
define the valid-source indicator

\[
v_{l,i}(x)=\mathbf1\!\left[
D^{\mathrm{drop}}_{l,i}(x)>\delta
\right].
\]

The present protocol uses \(\delta=10^{-5}\).

## 4. Coefficient-compensated substitution

For source (i\), representative (j\ne i\), and coefficient \(\alpha\), set

\[
g_i=0,\qquad g_j=\alpha,\qquad g_k=1\quad(k\notin\{i,j\}).
\]

Thus the original pair contribution

\[
c_{l,i}(x)+c_{l,j}(x)
\]

is replaced by

\[
\alpha c_{l,j}(x).
\]

Define the raw replacement error

\[
D_{l,i\rightarrow j}(x;\alpha)=D_l(x;g^{i\rightarrow j,\alpha}).
\]

For a fixed coefficient grid \(\mathcal A\),

\[
D^*_{l,i\rightarrow j}(x)
=\min_{\alpha\in\mathcal A}D_{l,i\rightarrow j}(x;\alpha),
\]

\[
\alpha^*_{l,i\rightarrow j}(x)
=\arg\min_{\alpha\in\mathcal A}D_{l,i\rightarrow j}(x;\alpha).
\]

The current evolution protocol uses 16 uniformly spaced coefficients on
\([0,3]\), and stores the entire error cube, not only its minimum.

## 5. Conditional Functional Substitutability

The directed, input-conditional CFS score is

\[
S_{l,i\rightarrow j}(x)
=1-
\frac{D^*_{l,i\rightarrow j}(x)}
{\max(D^{\mathrm{drop}}_{l,i}(x),\varepsilon)}.
\]

Here \(\varepsilon\) is only a numerical guard; scientific summaries additionally
apply the valid-source filter above.

Interpretation:

- \(S=1\): representative (j) perfectly recovers the deletion effect of (i);
- \(S=0\): substitution is no better than deleting (i);
- \(S<0\): substitution is worse than deletion;
- larger (S\): greater conditional substitutability.

CFS is generally directed:

\[
S_{i\rightarrow j}(x)\ne S_{j\rightarrow i}(x),
\]

and need not be transitive. It therefore defines a directed substitution graph,
not automatically an equivalence relation or quotient space.

## 6. Oracle recovery

For each valid source, its best available representative is

\[
j^*_{l,i}(x)=\arg\max_{j\ne i}S_{l,i\rightarrow j}(x),
\]

with source-level Oracle recovery

\[
O_{l,i}(x)=\max_{j\ne i}S_{l,i\rightarrow j}(x).
\]

The per-layer Raw Oracle is

\[
O_l=
\frac{
\sum_{x,i}v_{l,i}(x)O_{l,i}(x)
}{
\sum_{x,i}v_{l,i}(x)
}.
\]

Raw Oracle increases with candidate-set size, so it must not be used alone for
cross-model comparisons when the number of heads differs.

## 7. Matched-best-of-\(k\)

Matched-\(k\) controls the number of available representatives. For a source,
uniformly sample without replacement a candidate subset

\[
R_k\subseteq\{1,\ldots,H\}\setminus\{i\},\qquad |R_k|=k,
\]

and define

\[
M^{(k)}_{l,i}(x)
=\mathbb E_{R_k}\left[\max_{j\in R_k}S_{l,i\rightarrow j}(x)\right].
\]

This expectation is computed exactly from ordered candidate scores. If
\(s_{(1)}\le\cdots\le s_{(n)}\), where \(n=H-1\), then

\[
M^{(k)}=
\sum_{r=k}^{n}
\frac{\binom{r-1}{k-1}}{\binom{n}{k}}
s_{(r)}.
\]

Matched-1 is the mean candidate recovery; Matched-\((H-1)\) equals Raw Oracle.
The evolution database reports Matched-1, 2, and 4.

## 8. Absolute replacement and weighted recovery

The best absolute replacement KL is

\[
B_{l,i}(x)=\min_{j\ne i}D^*_{l,i\rightarrow j}(x).
\]

Mean deletion KL and mean best-substitution KL are reported jointly:

\[
\overline D_l^{\mathrm{drop}}
=\mathbb E[D^{\mathrm{drop}}_{l,i}(x)\mid v_{l,i}(x)=1],
\]

\[
\overline B_l
=\mathbb E[B_{l,i}(x)\mid v_{l,i}(x)=1].
\]

Their joint reporting distinguishes genuine reorganization from the degenerate
case in which every head simply becomes weak.

KL-weighted recovery first aggregates recovered error within each input:

\[
W_l(x)=
\frac{
\sum_i\left(D^{\mathrm{drop}}_{l,i}(x)-B_{l,i}(x)\right)
}{
\sum_iD^{\mathrm{drop}}_{l,i}(x)
},
\qquad
W_l=\mathbb E_x[W_l(x)].
\]

A positive-only variant replaces each numerator term by
\(\max(D^{\mathrm{drop}}-B,0)\).

## 9. Threshold graph and minimum functional cover

For threshold \(\tau\), define the directed substitution adjacency

\[
A^{(\tau)}_{l,ij}(x)
=\mathbf1[i\ne j]\,v_{l,i}(x)\,
\mathbf1[S_{l,i\rightarrow j}(x)>\tau].
\]

A retained representative set \(R\subseteq\{1,\ldots,H\}\) covers valid source
\(i\) when either (i\in R\), or some (j\in R\) satisfies
\(S_{i\rightarrow j}>\tau\). The exact minimum cover is

\[
C_l^{(\tau)}(x)
=\min_{R\subseteq[H]}|R|
\quad\text{s.t. every valid source is covered.}
\]

For a single input, \(C_l^{(\tau)}(x)\) is always an integer. A reported mean
cover

\[
\overline C_l^{(\tau)}=\mathbb E_x[C_l^{(\tau)}(x)]
\]

need not be an integer. The normalized quantity is

\[
\operatorname{Cover/Head}_l^{(\tau)}
=\frac{\overline C_l^{(\tau)}}{H}.
\]

Lower Cover/Head means that fewer representatives cover the valid functional
branches at that threshold; it indicates greater functional compression, not
automatically better task performance.

The evolution database uses \(\tau\in\{0.25,0.5,0.75\}\).

## 10. Directed spectral effective rank

Construct a nonnegative affinity matrix for each input:

\[
F_{ij}(x)=
\begin{cases}
\operatorname{clip}(S_{i\rightarrow j}(x),0,1),&i\ne j,\ v_i=1,\\
1,&i=j,\ v_i=1,\\
0,&v_i=0.
\end{cases}
\]

Let \(\sigma_1,\ldots,\sigma_H\) be its singular values and

\[
p_r=\frac{\sigma_r}{\sum_q\sigma_q}.
\]

The directed spectral effective rank is

\[
R_{\mathrm{eff}}(x)
=\exp\!\left(-\sum_{r:p_r>0}p_r\log p_r\right),
\]

with normalized rank

\[
\operatorname{Rank/Head}=\frac{\mathbb E_x[R_{\mathrm{eff}}(x)]}{H}.
\]

Singular values allow directed CFS without symmetrizing it. This is a
descriptive topology statistic, not a proof of the network's true functional
dimension.

## 11. Representative concentration and entropy

For valid sources in one input, count how often each head is selected as the
best representative:

\[
n_j(x)=\sum_i v_i(x)\mathbf1[j^*_i(x)=j],
\qquad
P_j(x)=\frac{n_j(x)}{\sum_k n_k(x)}.
\]

Representative concentration is

\[
\operatorname{Conc}(x)=\max_jP_j(x),
\]

and normalized representative entropy is

\[
\operatorname{Ent}(x)
=-\frac{\sum_{j:P_j>0}P_j(x)\log P_j(x)}{\log H}.
\]

High concentration/low entropy indicates that many sources point to a small
number of representative hubs.

## 12. Head energy and current-space controls

The head-energy proxy used in the current implementation is

\[
E_{l,i}(x)=\|c_{l,i}(x)\|_2,
\]

after flattening token and channel dimensions. It is reported to detect whether
CFS changes arise from functional reorganization or global head weakening.

Current-space pairwise distance is

\[
d^{\mathrm{L2}}_{ij}(x)
=\frac{\|c_i(x)-c_j(x)\|_2}
{\|c_i(x)\|_2+\|c_j(x)\|_2},
\]

and cosine distance is

\[
d^{\cos}_{ij}(x)
=1-\frac{\langle c_i,c_j\rangle}{\|c_i\|_2\|c_j\|_2}.
\]

These are controls for representation similarity; neither uses downstream
intervention.

The scalar Taylor importance baseline uses a differentiable head gate:

\[
T_{l,i}(x)=
\left|g_i\frac{\partial\mathcal L(x)}{\partial g_i}\right|_{g_i=1}.
\]

Taylor measures first-order source importance, not pairwise substitutability.

## 13. Pair deletion redundancy (auxiliary analysis)

For completeness, define joint deletion error

\[
D^{\mathrm{drop}}_{l,ij}(x)=D_l(x;g^{\setminus\{i,j\}}).
\]

The pattern

\[
D_i\le\epsilon,\qquad D_j\le\epsilon,
\qquad D_{ij}>\tau
\]

indicates set-level deletion redundancy: either branch can be removed, but not
both. A super-additivity statistic is

\[
\Delta_{ij}=D_{ij}-(D_i+D_j).
\]

This is related to branch interaction but is not the primary CFS definition.

## 14. Aggregation rules and current fixed protocol

The conditional score must be computed before dataset averaging:

\[
\mathbb E_x\left[
1-\frac{D^*_{i\rightarrow j}(x)}{D_i^{\mathrm{drop}}(x)}
\right]
\ne
1-\frac{\mathbb E_x[D^*_{i\rightarrow j}(x)]}
{\mathbb E_x[D_i^{\mathrm{drop}}(x)]}.
\]

The implementation uses the left-hand side. Macro values across layers are
equal-weight means of layer-level summaries, not a silent pooling of all
sources.

The current ViT-S training-evolution protocol is:

- epochs 1--100;
- layers 3, 7, and 11 (zero-based);
- the same frozen 512-image probe set at every epoch;
- 6 trained heads per layer;
- 16 alpha values uniformly spaced on \([0,3]\);
- (T=1\);
- valid-source threshold \(\delta=10^{-5}\);
- cover thresholds (0.25,0.5,0.75\);
- raw per-input matrices retained before any aggregation.

## 15. Direction of interpretation

- Higher Raw Oracle / Matched-\(k\): more deletion effect can be recovered by
  alternative representatives.
- Lower Cover/Head: stronger functional compression at the chosen threshold.
- Higher deletion KL: larger marginal effect of an individual source head.
- Higher Rank/Head: less spectrally concentrated directed substitution
  topology under this particular descriptive construction.
- None of these quantities alone means a model is better. Accuracy/loss,
  deletion effect, absolute replacement KL, CFS, cover, and rank must be
  interpreted jointly.
