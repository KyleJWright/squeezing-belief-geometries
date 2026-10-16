# Theory: belief planes, LayerNorm and Grassmannian packing

The mathematics behind [the README](README.md): what a belief plane is, which space the model can read,
why a model of width $`d`$ can only produce a $`(d-2)`$-parameter family of predictions, how overlaps between
planes are measured, how curvature is measured with a Gauss map, the Grassmannian packing problem, and why
the softmax favours linear structure. The
README's [theory in brief](README.md#theory-in-brief) summarises it and compares its predictions with what
was observed.


## Belief planes

A Mess3 process has three hidden states, so its belief $`\eta`$ lies in the triangle
$`\Delta^2 = \{\eta \in \mathbb{R}^3_{\ge 0} : \textstyle\sum_i \eta_i = 1\}`$. This lies in the affine plane
$`\sum_i \eta_i = 1`$ in $`\mathbb{R}^3`$, whose direction space (the tangent space at every interior point) is
the 2-plane $`T = \{v \in \mathbb{R}^3 : \sum_i v_i = 0\}`$. Let $`c = (\tfrac13, \tfrac13, \tfrac13)`$ be the
centre of the triangle; then $`\eta - c \in T`$ for every belief. `capacity.py` uses an orthonormal basis
$`E`$ of $`T`$, so a belief has tangent coordinates $`u = E^\top(\eta - c) \in \mathbb{R}^2`$.

Following Shai et al., the residual stream embeds beliefs (approximately) affinely: for process $`k`$,

```math
a \;\approx\; m_k + M_k(\eta - c), \qquad M_k : T \to \mathbb{R}^d.
```

The **belief plane** of process $`k`$ is $`V_k = \mathrm{im} M_k`$, a point of the Grassmannian
$`\mathrm{Gr}(2, d)`$, and its **belief patch** is the affine image $`m_k + M_k(\Delta^2 - c)`$. `capacity.py`
estimates $`M_k`$ by regressing activations on beliefs.

## Which space: LayerNorm

Every read of the residual stream goes through a LayerNorm. For an activation $`a \in \mathbb{R}^d`$,

```math
\mathrm{LN}(a) = \gamma \odot \frac{\sqrt{d}\, P a}{\lVert P a \rVert} + b, \qquad P = I - \tfrac{1}{d}\mathbf{1}\mathbf{1}^\top,
```

where $`\mathbf{1} = (1, \dots, 1) \in \mathbb{R}^d`$, so $`P`$ is the orthogonal projector onto
$`\mathbf{1}^\perp = \{z : \sum_i z_i = 0\}`$ (it subtracts the mean coordinate). The learned vectors
$`\gamma, b \in \mathbb{R}^d`$ are the scale and bias, and $`\odot`$ is the coordinatewise product,
$`(\gamma \odot z)_i = \gamma_i z_i`$, i.e. $`\gamma \odot z = \mathrm{diag}(\gamma)\, z`$.
(The small $`\epsilon`$ added to the norm in practice is ignored.)

Two consequences follow.

1. **The readable space is $`\mathbf{1}^\perp \cong \mathbb{R}^{d-1}`$.** The component of $`a`$ along
   $`\mathbf{1}`$ is discarded, so the planes that matter live in $`\mathrm{Gr}(2, d-1)`$; they are measured
   after projecting by $`P`$.
2. **The readout sees a sphere of dimension $`d-2`$.** $`\sqrt{d}\,Pa/\lVert Pa \rVert`$ lies on the sphere of radius
   $`\sqrt d`$ in $`\mathbf{1}^\perp`$, a manifold of dimension $`d-2`$, which $`\mathrm{diag}(\gamma)`$ and $`b`$
   map to an ellipsoid (degenerate if some $`\gamma_i = 0`$). The logits are linear in this point, so the
   next-token distributions a model of width $`d`$ can produce form the image of a smooth map from a
   manifold of dimension $`d-2`$.

For Mess3 (parameters $`x, \alpha`$, with $`y = 1 - 2x`$ and $`\beta = (1-\alpha)/2`$ as in `processes.py`)
the next-token distribution, as a row vector, is $`p(\cdot \mid \eta) = \eta N`$ with
$`N_{ik} = \Pr(\text{token } k \mid \text{state } i)`$. It is an injective affine function of $`\eta`$
($`N`$ is invertible whenever $`\alpha \ne \beta`$ and $`x \ne y`$). So over a set of beliefs $`B \subset \Delta^2`$,
an optimal predictor's outputs form an affine copy of $`B`$. The output head is a smooth map $`g`$ from the
sphere $`S^{d-2}`$ to distributions, $`g(S^{d-2})`$ has Hausdorff dimension at most $`d-2`$, and an affine
bijection preserves Hausdorff dimension. Exact optimal prediction on $`B`$ therefore needs

```math
d - 2 \;\ge\; \dim_H B .
```

No assumption is needed about how the network maps prefixes to the sphere; only the output head has to be
smooth. Three caveats decide how much this says.

- **Finite context.** With a context of 10, $`B`$ is a finite set, of dimension 0, and the bound is empty:
  a curve can pass through any finite set of points. It bites only in the limit of long contexts, where
  $`B`$ fills out the fractal attractor. For finite contexts it is a heuristic: a map with bounded weights
  cannot wiggle enough to thread a curve through tens of thousands of nearby points.
- **Fractal dimension.** The attractor's Hausdorff dimension lies between 0 and 2. The threshold $`d \ge 4`$
  assumes it exceeds 1, which the images suggest but which is not computed here.
- **Two processes.** The table treats each belief set as 2-dimensional, as a filled region would be. For the
  fractals themselves the sparse bound is $`d - 2 \ge \max(\dim_H B_A, \dim_H B_B)`$ and the dense one is
  $`d - 2 \ge \dim_H (B_A \times B_B) \ge \dim_H B_A + \dim_H B_B`$, which gives $`d \ge 6`$ only if the two
  dimensions sum to more than 3.

| Setting | Optimal predictions | Dimension (filled regions) | Readout bound |
| --- | --- | --- | --- |
| One process | $`\{\eta N\}`$ | 2 | $`d \ge 4`$ |
| Two processes, sparse | union of two 2-dimensional families | 2 | $`d \ge 4`$ |
| Two processes, dense | products $`p_A(t_A)\,p_B(t_B)`$ over token pairs $`(t_A, t_B)`$ | 4 | $`d \ge 6`$ |

This is a necessary condition for *exact* prediction only: below it the model must approximate, and
the measured loss shows the cost.

At $`d = 2`$ the sphere is $`S^0`$, two points, and the normalisation $`z \mapsto z/\lvert z\rvert`$ on the line
$`\mathbf{1}^\perp`$ is a sign function. Its derivative vanishes away from 0, so no gradient passes back
through any LayerNorm.

## Overlap, principal angles and chordal distance

For planes $`V, W \in \mathrm{Gr}(2, n)`$ with orthonormal bases $`Q_V, Q_W`$, the singular values of
$`Q_V^\top Q_W`$ are $`\cos\theta_1 \ge \cos\theta_2`$, the cosines of the principal angles. The overlap used
throughout is

```math
\mathrm{ov}(V, W) = \tfrac12 \lVert Q_V^\top Q_W \rVert_F^2 = \tfrac12\left(\cos^2\theta_1 + \cos^2\theta_2\right) = \tfrac12 \mathrm{tr}(P_V P_W),
```

and the chordal distance is $`d_c^2 = \sin^2\theta_1 + \sin^2\theta_2 = \tfrac12\lVert P_V - P_W\rVert_F^2 = 2\,(1 - \mathrm{ov})`$.
For independent uniformly random planes, $`\mathbb{E}\,\mathrm{tr}(P_V P_W) = 4/n`$, so the mean overlap
is $`2/n`$.

## What is measured

The numbers in the README come from these choices, stated explicitly because angles are not
canonical:

1. **Space.** The final residual stream (the input to the output head, before its LayerNorm), with
   the all-ones direction projected out by $`P`$: the readable space $`\mathbf{1}^\perp \cong \mathbb{R}^{d-1}`$.
2. **Planes.** For each process, the column span of $`P M_k`$, where $`M_k`$ comes from a least-squares fit
   of activations on beliefs. These are the *linear direction spaces* of the belief patches: points of
   $`\mathrm{Gr}(2, d-1)`$. Offsets $`m_k`$ are measured separately, as the distance between process centres
   divided by the within-process spread.
3. **Metric.** Overlaps need an inner product, and the Euclidean one on the residual stream is a
   convention: the Grassmannian is invariant under $`GL`$, but principal angles are invariant only under
   orthogonal maps. So overlaps are reported in two metrics on $`\mathbf{1}^\perp`$: the **Euclidean** one,
   and the **γ-metric**
   $`\langle z, w\rangle_\gamma = z^\top \mathrm{diag}(\gamma)^2\, w = (\gamma \odot z)\cdot(\gamma \odot w)`$,
   where $`\gamma`$ is the output head's LayerNorm scale. The γ-metric is the geometry the readout's first
   linear step imposes. Conclusions are drawn only where the two agree.
4. **Not modelled.** The normalisation $`a \mapsto \sqrt d\,Pa/\lVert Pa\rVert`$ maps a flat patch not
   through the origin to a curved patch on the sphere, whose tangent plane varies from point to point.
   The planes are measured before this step. The dimension counts above do not depend on any of these
   choices.

The readout referred to throughout is the output head,
$`\text{logits} = W_U\,\mathrm{LN}(a) + b_U`$, followed by a softmax.

## Curvature: the Gauss map

The belief-plane picture is an affine approximation. Write the embedding of beliefs as a map
$`f : \Delta^2 \to \mathbf{1}^\perp`$, $`u \mapsto \mathbb{E}[a \mid u]`$ (in tangent coordinates $`u`$), and
suppose it is smooth. Its **Gauss map** sends each point to its tangent plane,

```math
\mathcal{G} : \Delta^2 \to \mathrm{Gr}(2, d-1), \qquad u \mapsto \mathrm{im}\, Df(u).
```

The patch is (an open piece of) a plane exactly when $`\mathcal{G}`$ is constant, and the variation of
$`\mathcal{G}`$ measures curvature. Two features make it a good measure here:

- **It ignores parametrisation.** Composing $`f`$ with a diffeomorphism of the triangle, for instance using
  optimal logits $`\log(\eta N)`$ instead of beliefs as coordinates, leaves the image of $`Df`$ unchanged.
  Bending *within* the plane is invisible to it; only bending *out of* the plane counts.
- **It lives where everything else does.** The overlap between local and global planes uses the same
  principal angles as the overlap between two processes' planes.

A linear decoder is blind to this. If $`f(u) = m + Mu + h(u)\,n`$ with $`n \perp \mathrm{im}\, M`$, the
decoder $`M^{+}`$ recovers $`u`$ exactly for any $`h`$, while $`Df(u) = M + n\, \nabla h(u)^\top`$ tilts
out of $`\mathrm{im}\, M`$ wherever $`\nabla h \ne 0`$.

`gauss_map.py` estimates $`Df`$ at anchor beliefs by local least squares over nearby beliefs, so it sees
curvature on the scale of the neighbourhood. With finitely many beliefs at context 10, $`f`$ is only defined on
a finite set, and $`\mathbb{E}[a \mid u]`$ is estimated by averaging over nearby beliefs. Wrinkles finer than
the neighbourhood are therefore not resolved, and noise sets a floor, estimated from a synthetic flat patch.

## The packing problem and its bounds

**Grassmannian packing** (Conway, Hardin & Sloane 1996): place $`K`$ planes in $`\mathbb{R}^n`$ to minimise
the largest pairwise overlap, equivalently to maximise the smallest chordal distance.

- **Orthogonality is possible iff $`2K \le n`$.** For two planes, $`n \ge 4`$.
- **Two planes in $`\mathbb{R}^3`$** must meet in a line, since $`\dim(V \cap W) \ge 2 + 2 - 3 = 1`$. So
  $`\theta_1 = 0`$, and the best is $`\theta_2 = \pi/2`$, giving $`\mathrm{ov} = \tfrac12`$.
- **The simplex (Rankin) bound.** Let $`S = \sum_k P_{V_k}`$, so $`\mathrm{tr} S = 2K`$ and
  $`\mathrm{rank} S \le n`$. By Cauchy–Schwarz on its eigenvalues, $`\lVert S\rVert_F^2 \ge (2K)^2/n`$.
  Expanding $`\lVert S \rVert_F^2 = 2K + \sum_{i \ne j}\mathrm{tr}(P_iP_j)`$ gives

  ```math
  \max_{i\ne j} \mathrm{ov}(V_i, V_j) \;\ge\; \mathrm{mean}_{i\ne j} \mathrm{ov}(V_i, V_j) \;\ge\; \frac{2K/n - 1}{K - 1}.
  ```

  Equality needs $`S = \tfrac{2K}{n} I`$ (a tight fusion frame) *and* all pairwise overlaps equal. Three
  planes in $`\mathbb{R}^4`$ attain it (overlap $`\tfrac14`$). Two planes in $`\mathbb{R}^3`$ do not: the
  bound gives $`\tfrac13`$, but the true optimum is $`\tfrac12`$, because the shared line forces unequal
  principal angles.

`packing.py` computes the optimum numerically and reproduces both cases.

## Why packing matters in the dense regime, and not in the sparse one

**Dense: simultaneous decoding.** Both beliefs are active at once, $`a = m + M_A u_A + M_B u_B`$. Decoding
both linearly needs $`[M_A\; M_B]`$ to be injective, i.e. $`V_A \cap V_B = 0`$: linear independence, not
orthogonality. Orthogonality matters for noise. The Gram matrix of $`[Q_A\; Q_B]`$ is
$`\begin{pmatrix} I & C \\ C^\top & I \end{pmatrix}`$ with $`C = Q_A^\top Q_B`$, whose eigenvalues are
$`1 \pm \cos\theta_i`$. Least-squares decoding therefore amplifies noise in the activations by up to

```math
\frac{1}{\sqrt{1 - \cos\theta_1}},
```

which is smallest when the planes are as far apart as possible. A model that must read both beliefs
through noise is pushed towards the optimal packing.

**Sparse: one patch at a time.** Only one process is active, so the model must (i) tell which patch the
activation is in, and (ii) decode the belief within it. Two compact convex patches are disjoint exactly
when a hyperplane separates them, and nothing in this constrains the directions $`V_A, V_B`$. Two
triangles can share a plane and sit side by side. The relevant object is an arrangement of *affine*
patches, not a packing of linear subspaces, and the Grassmannian overlap can be large at no cost.

**Sparse with shared tokens: a common quotient.** With one set of tokens, the optimal prediction for either
process is a distribution over the same 3 tokens: $`p = \eta_A N_A`$ or $`p = \eta_B N_B`$, both in the same
2-dimensional family. If the final layer carries $`p`$ (or any affine image of it), both beliefs map into a
single plane and the patches overlap, with no need to tell the processes apart there. Separate planes would
be needed only if something downstream had to read the two beliefs separately, and at the final layer
nothing does.

## The softmax, and why it favours linear structure

The output is $`\mathrm{softmax}(\ell)`$, a nonlinear function of the logits $`\ell`$. Does this warp the
linear picture? It acts after the residual stream, so it cannot bend the planes measured there. What it
changes is which residual geometry is *useful*, and the natural linear target is the logits, not the
probabilities. Logits are defined up to an additive constant, so take them centred.

**One process.** With three tokens, centred logits live in a plane, and the optimal ones form
$`\eta \mapsto \log(\eta N)`$ (centred). This is a smooth, injective map of the triangle onto a region of that
plane. So the softmax curves coordinates *within* a plane; it does not take the patch out of it.

**Dense: products become sums.** For two processes run side by side, each token is a pair
$`(t_A, t_B)`$ of sub-tokens, so distributions and logits are functions on the 9 pairs. The optimal
distribution is the product $`p(t_A, t_B) = p_A(t_A)\, p_B(t_B)`$, so its logits are a sum,

```math
\log p(t_A, t_B) = \log p_A(t_A) + \log p_B(t_B).
```

Conversely, if $`\ell(t_A, t_B) = \ell_A(t_A) + \ell_B(t_B)`$, then
$`\mathrm{softmax}(\ell)(t_A, t_B) = \mathrm{softmax}(\ell_A)(t_A)\,\mathrm{softmax}(\ell_B)(t_B)`$.

Centred functions of $`t_A`$ alone and of $`t_B`$ alone span two 2-dimensional subspaces of the 8-dimensional
centred logit space, and these subspaces are orthogonal. So the optimal logit vector is *exactly* a sum of
an A-part and a B-part in the orthogonal direct sum $`L_A \oplus L_B`$ of those two subspaces. The softmax
turns the multiplicative structure of independent processes into additive, linear structure, which a
linear readout produces most simply from two independent planes in the residual stream: the setting of
the packing problem.

**Sparse: the nonlinearity does the separating.** With one process active, the optimal distribution puts
probability zero on the other process's tokens, which needs their logits to be very negative. A large
offset along a direction that separates the two token blocks achieves this, and the softmax saturates on
the far side. This is the offset separation measured in [README section 4](README.md#4-geometry-of-the-belief-patches),
and it is why sparse planes can coincide at no cost.

**Check.** `capacity.py` refits every plane using the centred optimal logits $`\log(\eta N)`$ as coordinates
instead of $`\eta`$. If the belief-dependent part of the activations is a curved patch in a fixed plane, both
fits find that plane. They do: the overlap between the two fits of the same process (`plane_agreement`) is
at least 0.999 for every represented process. This shows the belief-dependent part lies in a plane; it
cannot say whether the model "uses" beliefs or logits as coordinates on it.

## References

- A. Shai, S. Marzen, L. Teixeira, A. Gietelink Oldenziel, P. Riechers.
  [Transformers Represent Belief State Geometry in their Residual Stream](https://arxiv.org/abs/2405.15943).
  NeurIPS 2024.
- J. H. Conway, R. H. Hardin, N. J. A. Sloane. Packing lines, planes, etc.: packings in Grassmannian spaces.
  *Experimental Mathematics* 5(2), 139–159, 1996.
- R. A. Rankin. The closest packing of spherical caps in n dimensions. *Proceedings of the Glasgow
  Mathematical Association* 2, 139–144, 1955.
