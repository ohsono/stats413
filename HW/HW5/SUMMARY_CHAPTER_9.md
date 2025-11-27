# Chapter 9 – Trees and Boosting

## 9.0 Framing
Chapter 9 recasts regression trees and boosting as alternative forms of the same incremental-improvement principle that drives gradient-descent training of neural networks. The chapter progresses from decision trees through L2 boosting and XGBoost, interprets boosting as surrogate-loss minimization with implicit regularization, contrasts AdaBoost with modern gradient boosting, and ends with Random Forests as a parallel ensemble counterpart.

## 9.1 Incremental Viewpoint
Gradient descent updates parameters (Eq. 9.1), trees refine partitions by recursive splits, and boosting adds trees in an additive model. All three minimize objectives of the form $(\sum_i L(y_i,f(x_i))+\text{complexity}(f)$) (Eq. 9.2) and embody bias→variance trade-offs as the number or size of increments grows.

## 9.2 Decision Trees
- Trees partition feature space via binary splits guided by purity metrics such as misclassification rate, Gini $\sum_k p_k(1-p_k)$
(Eq. 9.4) or entropy \(-\sum_k p_k \log p_k\) (Eq. 9.5).
- The split at region \(R\) and feature threshold \(s\) maximizes impurity decrease \(\Delta I\) (Eq. 9.7). For regression, each leaf predicts the average response (Eq. 9.8) and minimizes squared error (Eq. 9.9).

## 9.3 Regression Trees
- Represented as piecewise-constant functions \(f(x)=\sum_m c_m \mathbf{1}_{x\in R_m}\) (Eq. 9.10), fitted by minimizing squared error plus complexity penalties (Eq. 9.11).
- Recursive binary splitting evaluates candidate feature-threshold pairs via RSS reduction (Eq. 9.12) and grows the tree until stopping rules (node size, impurity) are met.
- Cost-complexity pruning evaluates subtrees via validation or cross-validation to avoid overfitting (Eq. 9.13).

## 9.4 L2 Boosting / Gradient Boosted Trees
- Boosting constructs \(f_M(x)=\sum_{m=1}^M h_m(x)\) (Eq. 9.14) where each tree \(h_m\) has leaf regions \(R_{mj}\) with predictions \(c_{mj}\) (Eq. 9.15).
- Adding a tree solves a regularized least squares problem (Eq. 9.16), yielding optimal leaf values \(c^*_{L,R}=\frac{\sum_{i\in R} r_i}{|R|+\lambda}-\gamma\) (Eq. 9.19). Weighted variants handle observation weights \(w_i\) (Eqs. 9.20–9.23).
- Algorithm 9.4 collects residuals (negative gradients), fits a tree with shrinkage (learning rate), and updates the ensemble.
- The procedure is equivalent to functional gradient descent, and setting \(\lambda,\gamma\) controls tree complexity.

## 9.5 XGBoost for Logistic Regression
- Logistic models use \(p=1/(1+e^{-f(x)})\) (Eq. 9.24) with additive trees (Eq. 9.25). Logistic loss \(l(y,f)= -y\log p -(1-y)\log(1-p)\) (Eq. 9.26) yields gradients \(g=p-y\) (Eq. 9.30) and Hessians \(h=p(1-p)\) (Eq. 9.31).
- Adding tree \(h_m\) minimizes the second-order surrogate \(\sum_i [g_i h(x_i)+\tfrac12 h_i h(x_i)^2]+\Omega(h)\) (Eq. 9.40) leading to a weighted least squares fit with weights \(w_i=h_i\) and working responses \(z_i=-g_i/h_i\) (Eq. 9.41).
- Split scoring becomes \(\text{Gain} = \frac{(\sum_{i\in R_L} g_i)^2}{\sum_{i\in R_L} h_i+\lambda} + \frac{(\sum_{i\in R_R} g_i)^2}{\sum_{i\in R_R} h_i+\lambda} - \frac{(\sum_{i\in R} g_i)^2}{\sum_{i\in R} h_i+\lambda} - \gamma\) (Eqs. 9.34–9.35).
- Multiple interpretations strengthen intuition: Newton steps in function space, geometric “golf” analogy, and connections to back-propagation and IRLS (Eqs. 9.36–9.41).

## 9.6 Surrogate Losses & Incremental Learning
- Gradient descent minimizes quadratic surrogates (Eqs. 9.42–9.43). Boosting minimises surrogate losses such as exponential (AdaBoost), logistic (LogitBoost/XGBoost), or squared error (L2Boosting) (Eqs. 9.44–9.46).
- XGBoost’s surrogate (Eq. 9.47) uses gradients and Hessians, leading to efficient tree search and regularization. Good surrogates upper-bound the true loss, match at current iterate, and are easily optimized.

## 9.7 Lazy/Implicit Regularization
- Gradient flow in function space (Eq. 9.50) illustrates “lazy” trajectories: small learning rates (Eq. 9.51) prioritize low-frequency/global structure first (Eq. 9.52) and control complexity roughly proportional to \(M\eta\) (Eq. 9.53).
- This implicit bias parallels deep networks: early stopping acts as regularization, incremental updates bound changes (Eqs. 9.54–9.56), and spectral bias emerges naturally.

## 9.8 AdaBoost vs XGBoost
- Loss functions: AdaBoost’s exponential loss (Eq. 9.78) vs XGBoost’s logistic (Eq. 9.79) highlight sensitivity to outliers and Hessian availability.
- Base learners: AdaBoost typically uses shallow stumps with ±1 outputs; XGBoost uses deeper regression trees with real-valued leaf outputs.
- Update rules: AdaBoost multiplies observation weights, while XGBoost performs Newton-style gradient/Hessian updates. Tables 9.2–9.3 summarize differences in regularization, parallelization, memory, and use cases.

## 9.9 Random Forests
- Ensemble of independently grown trees: \(f_M(x)=\frac{1}{M}\sum_{m=1}^M T_m(x)\) (Eq. 9.80).
- Randomness sources: bootstrap sampling (Eq. 9.81) and feature subsampling with \(k\approx\sqrt{p}\) (Eq. 9.82). Construction algorithm (Alg. 11) evaluates splits using RSS (Eq. 9.83).
- Statistical properties: variance decomposes via tree correlation (Eq. 9.84); out-of-bag prediction (Eq. 9.85) yields unbiased error estimates (Eq. 9.86).
- Variable importance: mean decrease in impurity (Eq. 9.87) and mean decrease in accuracy via permutation (Eq. 9.88).
- Consistency and convergence results (Eqs. 9.89–9.90) hold under depth/size conditions. Table 9.4 compares Random Forest, AdaBoost, and XGBoost.

---
Chapter 9, through a unified lens of incremental improvement and surrogate optimization, connects decision trees, boosting, and Random Forests, explaining why modern gradient boosting (e.g., XGBoost) achieves strong performance while still linking back to classical AdaBoost and bagging methods.
