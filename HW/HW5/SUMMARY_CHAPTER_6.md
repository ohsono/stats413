# Chapter 6 – Diffusion Models

## 6.0 Framing
Chapter 6 develops diffusion models as probability “time machines.” Starting from population-based intuitions for Bayes’ rule, it shows how adding Gaussian noise (forward diffusion) and following the score function (reverse denoising) form perfectly reversible steps. The chapter then scales this intuition to long trajectories, neural score parameterization, variance reduction, maximum-likelihood training, and continuous-time limits, culminating in flow-matching and variance-scheduled samplers used in modern image generators.

## 6.1 Probability Preliminaries
A billion-person toy example grounds discrete rules like chain, marginalization, and conditioning \((6.1)\). In continuous form the same rules connect densities via integrations, motivating why the backward conditional \(p(x|y)\) is the lever for reversing diffusion.

## 6.2 Single-Step Noising and Denoising
Forward noise adds \(e \sim \mathcal{N}(0,\sigma^2)\) to give \(y = x + e\). Bayes plus a first-order Taylor expansion shows the posterior is \(p(x|y) = \mathcal{N}(y + \sigma^2 \nabla \log p(y), \sigma^2)\), so the mean nudges noisy data toward higher density using the score ∇log p. Re-applying this step exactly counteracts the spreading induced by noise, making diffusion reversible. Deterministic denoising requires a half-step \(x = y + \tfrac{1}{2}\sigma^2 ∇\log p(y)\) to avoid over-concentration, while stochastic denoising samples the full posterior by adding fresh noise.

## 6.3 Trajectory-Based Modeling
Instead of modeling \(p(x_0)\) directly, diffusion augments data with trajectories \((x_0, …, x_T)\) that drift toward a tractable Gaussian endpoint \(x_T \sim \mathcal{N}(0, \sigma_T^2 I)\) \((6.1)\). Generation samples the endpoint and iteratively applies learned reverse kernels \(p_\theta(x_{t-1}|x_t)\), mirroring GPT’s autoregressive decomposition but along a synthetic time axis.

## 6.4 Gaussian Trajectories and Reverse Kernels
Forward steps use additive Gaussians \(x_t = x_{t-1} + e_t\) with \(e_t \sim \mathcal{N}(0, \sigma^2 I)\) \((6.8)-(6.12)\); after many steps \(x_T\) approaches a simple Gaussian regardless of \(x_0\). Taylor expansion plus Bayes gives \(p(x_{t-1}|x_t) \approx \mathcal{N}(x_t + \sigma^2 ∇\log p_{t-1}(x_t), \sigma^2 I)\) \((6.18)\). The necessity of Gaussian noise and small \(\sigma^2\) stems from high-dimensional tractability: only Gaussians admit closed-form accumulation and inversion.

## 6.5 Score-Based Parameterization
A single neural network \(s_\theta(x_t, t)\) predicts the score for every timestep: \(p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_t + \sigma^2 s_\theta(x_t, t), \sigma^2 I)\) \((6.21)\). Training minimizes \(\mathbb{E}\|x_{t-1} - (x_t + \sigma^2 s_\theta(x_t, t))\|^2\) \((6.22)\), or, via algebra, the clean-target loss \(\|x_0 - (x_t + t \sigma^2 s_\theta(x_t, t))\|^2\) \((6.23)\). Reparameterizing \(\epsilon_\theta = - t \sigma^2 s_\theta\) yields the noise-prediction loss \(\mathbb{E}\|\epsilon_t - \epsilon_\theta(x_t, t)\|^2\) \((6.25)\). UNets provide the backbone, with sinusoidal time embeddings \((6.27)\), ResNet blocks \((6.28)\), and optional attention \((6.29)\).

## 6.6 Variance Reduction
Rather than Monte Carlo over many \(x_{t-1}\) sharing the same \((x_0, x_t)\), the conditional mean \(\bar{x}_{t-1} = \mathbb{E}[x_{t-1}|x_0, x_t] = \tfrac{1}{t} x_0 + (1 - \tfrac{1}{t}) x_t\) \((6.32)-(6.34)\) serves as a deterministic target, eliminating inner-sample variance and yielding the clean-target objective \((6.23)\).

## 6.7 Denoising Auto-encoder and Vincent/Tweedie Identity
Vincent’s identity \(\mathbb{E}[x_0|x_t] = x_t + t\sigma^2 ∇\log p_t(x_t)\) \((6.35)-(6.41)\) proves that optimal denoisers produce the score. Thus diffusion training is equivalent to denoising auto-encoding with Gaussian corruption.

## 6.8 Noise-Prediction Implementation
Algorithm 4 trains \(\epsilon_\theta\) by sampling \(t\sim \mathcal{U}(1,T)\), corrupting data with \(\epsilon_t \sim \mathcal{N}(0, t \sigma^2 I)\), and minimizing MSE \((6.43)\). Algorithm 5 reverses diffusion by iteratively subtracting predicted noise and reintroducing Gaussian randomness.

## 6.9 Maximum Likelihood and KL Perspective
Trajectory likelihood factorizes as \(p(x_T) \prod_t p_\theta(x_{t-1}|x_t)\) \((6.50)-(6.53)\). Minimizing \(D_{\mathrm{KL}}(P_{\text{data}} \Vert P_\theta)\) decomposes into matching the terminal Gaussian and each reverse conditional \((6.56)-(6.66)\), yielding the same MSE objective but with a principled probabilistic justification.

## 6.10 Deterministic Sampling and the t−2 Phenomenon
Because \(x_{t-1} + e_t\) follows the correct posterior when \(e_t\) is fresh Gaussian noise, the deterministic update \(\tilde{x}_{t-1} = x_t + \sigma^2 ∇ \log p_t(x_t)\) already lands on the distribution two steps earlier. Hence the stochastic sampler effectively halves time, and pure ODE samplers become possible.

## 6.11 Continuous-Time Limits
Letting \(\Delta t = 1/T\) with \(\sigma^2/T\) variance per step gives the forward SDE \(dx = \sigma \, dw_t\) and reverse SDE \(dx = -\sigma^2 s_\theta(x,t) dt + \sigma \, dw_t\) \((6.72)-(6.75)\). The deterministic limit yields the probability-flow ODE \(dx/dt = -\tfrac{1}{2} \sigma^2 s_\theta(x,t)\) \((6.79)\). The “movie frame” analogy clarifies why \(\sqrt{\Delta t}\) scaling produces the characteristic zig-zag Brownian paths.

## 6.12 Langevin Dynamics vs Denoising
Forward diffusion is Brownian motion; Langevin dynamics adds score-driven drift with \(1/2\) coefficient to preserve the target distribution \((6.84)\). Denoising uses a coefficient of 1 (non-equilibrium) for SDE \((6.85)\) or \(1/2\) for the deterministic probability-flow ODE \((6.86)\), transporting \(p_t\) to \(p_{t-\Delta t}\).

## 6.13 General Drift
With a forward drift \(f(x,t)\) and time-varying noise \(\sigma_t\) \((6.87)\), the reverse SDE subtracts the drift and adds score guidance \((6.88)\), while the deterministic ODE uses the \(\tfrac{1}{2}\sigma_t^2\) factor \((6.89)\).

## 6.14 Random Drift vs Diffusion
If the random term scales with \(\Delta t\) instead of \(\sqrt{\Delta t}\) \((6.90)-(6.91)\), its variance vanishes in the limit by the law of large numbers, making the process equivalent to a deterministic ODE. Diffusion terms with \(\sqrt{\Delta t}\) accumulate finite variance via the central limit theorem.

## 6.15 Fokker–Planck View
Analyzing expectations of smooth test functions under the discrete updates recovers the Fokker–Planck equation \(\partial_t p = -\partial_x(f p) + \tfrac{1}{2} \partial_x^2(\sigma^2 p)\) \((6.96)-(6.99)\) for SDEs and the continuity equation \(\partial_t p = -\partial_x(v p)\) \((6.100)-(6.102)\) for deterministic flows, offering a complementary lens on the t−2 intuition.

## 6.16 Flow Matching with Straight Trajectories
Instead of stochastic diffusion, one can sample straight lines between data \(x_0 \sim p_{\text{data}}\) and Gaussian anchors \(x_1 \sim \mathcal{N}(0,I)\): \(x_t = (1-t) x_0 + t x_1\) \((6.109)\). Learning the velocity field \(v_\theta(x_t,t) \approx \mathbb{E}[x_0 - x_1 | x_t]\) \((6.110)-(6.112)\) recovers noise- and score-prediction objectives via simple rescalings \((6.113)-(6.117)\), connecting flow matching to classical diffusion training.

## 6.17 Variance Scheduling
Practical samplers scale states and add noise with \(x_t = \sqrt{\alpha_t} x_{t-1} + e_t, \ e_t \sim \mathcal{N}(0, \beta_t I)\) \((6.118)\). The cumulative product \(\bar{\alpha}_t = \prod_{i=1}^t \alpha_i\) yields a closed-form marginal \(q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)\) \((6.122)-(6.126)\). Training stays as simple MSE on \(\epsilon\), while sampling rescales by \(\alpha_t\) and injects fresh Gaussian noise, underpinning DDPM/DDIM implementations.

---
Chapter 6 unifies diffusion as reversible probabilistic transport, practical UNet-based score estimation, likelihood-based training, and continuous-time interpretations, while also highlighting deterministic flow-matching alternatives and variance scheduling used in state-of-the-art generative models.
