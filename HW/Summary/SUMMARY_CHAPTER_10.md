# Chapter 10 – Support Vector Machine

## 10.0 Framing
Chapter 10 recasts SVMs as a geometric max-margin classifier: project classes onto a direction, maximize their separation, and translate that intuition into convex optimization. The treatment proceeds from primal formulation through Lagrangian duality, interprets the dual as finding the minimum distance between class convex hulls, derives a practical dual coordinate-ascent algorithm, and generalizes everything via the kernel trick/RKHS theory.

## 10.1 Max-Margin Primal
For binary labels \(y_i\in\{-1,+1\}\), projecting onto \(u\) exposes extrema \(a^+=\min_{y_i=+1}\langle x_i,u\rangle\) and \(a^-=\max_{y_i=-1}\langle x_i,u\rangle\) (Eqs. 10.1–10.2); data are separable iff \(a^-<a^+\) (Eq. 10.3) and maximizing the margin \(a^+-a^-\) (Eq. 10.4) yields the standard primal \(\min_{w,b}\tfrac12\|w\|^2\) subject to \(y_i(\langle w,x_i\rangle+b)\ge1\) (Eqs. 10.10–10.11). Slack/hinge-loss extensions handle non-separable data, though these details are only alluded to here.

## 10.2 Lagrangian Min–Max
Introducing multipliers \(\alpha_i\ge0\) forms the Lagrangian \(L(w,b,\alpha)=\tfrac12\|w\|^2+\sum_i \alpha_i[1-y_i(\langle w,x_i\rangle+b)]\) (Eq. 10.14). The primal equals \(\min_{w,b}\max_{\alpha\ge0} L\) (Eqs. 10.15–10.16), and by convexity one can swap min/max (Eq. 10.24) because a saddle point satisfies \(L(w^*,b^*,\alpha)\le L(w^*,b^*,\alpha^*)\le L(w,b,\alpha^*)\). A game-theoretic view (Eq. 10.28) underscores why strong duality holds.

## 10.3 Dual and Geometric Interpretation
Eliminating \(w,b\) gives the dual max problem \(\sum_i \alpha_i - \tfrac12 \sum_{i,j} \alpha_i\alpha_j y_i y_j \langle x_i,x_j\rangle\) with constraints \(\alpha_i\ge0\) and \(\sum_i \alpha_i y_i=0\) (Eqs. 10.34–10.36). Geometrically, \(w\) points from the closest convex combination of positive examples \(\bar x_+\) to that of negatives \(\bar x_-\) (Eqs. 10.39–10.54). Support vectors are precisely the examples with \(\alpha_i>0\), enforced by KKT conditions (Eqs. 10.56–10.58); the margin equals \(2/\|w\| = 2/\|\bar x_+ - \bar x_-\|\) (Eq. 10.55).

## 10.4 Dual Coordinate Ascent
Assuming \(b=0\) for simplicity, each coordinate update optimizes \(f(\alpha_i)=\alpha_i - \tfrac12 Q_{ii}\alpha_i^2 - \alpha_i \sum_{j\ne i} y_i y_j Q_{ij} \alpha_j\) (Eq. 10.61) with gradient \(g_i=1-y_i\langle w,x_i\rangle\) (Eq. 10.64), yielding \(\alpha_i \leftarrow \max(0, y_i g_i/Q_{ii})\) (Eq. 10.65). Algorithm 12 maintains \(w=\sum_i \alpha_i y_i x_i\), updates coordinates sequentially, and checks convergence via maximum KKT violation (Eq. 10.66); “shrinking” skips coordinates already satisfying KKT conditions.

## 10.5 Kernel Trick and RKHS
Because the dual depends only on inner products, replacing \(\langle x_i,x_j\rangle\) with a kernel \(K(x_i,x_j)=\langle \Phi(x_i),\Phi(x_j)\rangle\) (Eq. 10.71) yields the kernelized dual (Eq. 10.74) and decision function \(f(x)=\sum_i \alpha_i y_i K(x_i,x)\) (Eq. 10.76). Common kernels include polynomial, RBF, and sigmoid (Eqs. 10.77–10.79); Mercer’s theorem (Eq. 10.80) ensures positive semidefiniteness. Kernelized coordinate ascent uses gradients \(g_i=1-y_i\sum_j \alpha_j y_j K(x_i,x_j)\) and updates \(\alpha_i\) via Eq. 10.82, with kernel caches and diagonal regularization for stability.

## 10.6 RKHS Perspective
Functions in an RKHS have the form \(f(x)=\sum_i \alpha_i K(x,x_i)\) with norm \(\|f\|_H^2=\sum_{i,j} \alpha_i \alpha_j K(x_i,x_j)\) (Eqs. 10.83–10.94). The reproducing property \(\langle f,K(\cdot,x)\rangle_H = f(x)\) links evaluation to inner products. This framework explains why SVM solutions depend only on support vectors and why regularizing \(\|f\|_H\) corresponds to maximizing margins in the implicit feature space.

---
Chapter 10 thus unifies the geometric margin picture, convex duality, algorithmic implementation, and kernel/RKHS theory into a single view of SVMs as sparse max-margin classifiers operating in potentially infinite-dimensional feature spaces.
