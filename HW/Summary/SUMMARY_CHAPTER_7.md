# Chapter 7 – VAE and GAN

## 7.0 Framing
Chapter 7 connects classical maximum-likelihood estimation to modern latent-variable generators. It motivates VAEs by viewing observed data as effects of latent causes, deriving the ELBO from joint KL divergence, and contrasting learned inference with the fixed forward process leveraged by diffusion models. The chapter then pivots to GANs, interpreting adversarial training as a game between generator and discriminator and culminating in the Wasserstein formulation.

## 7.1 Maximum Likelihood and KL Divergence
Independent samples \(x_1,\dots,x_n\sim p_{\text{data}}(x)\) yield the empirical log-likelihood \(L(\theta)=\frac{1}{n}\sum_i\log p_\theta(x_i)\to\mathbb{E}_{p_{\text{data}}}\log p_\theta\). The true distribution’s log-likelihood \(L(p_{\text{data}})=\mathbb{E}\log p_{\text{data}}=-H(p_{\text{data}})\) sets the entropy-dependent ceiling. Their gap is the marginal KL \(D_{\mathrm{KL}}(p_{\text{data}}\Vert p_\theta)=L(p_{\text{data}})-L(\theta)\), showing maximum likelihood as KL minimization. Geometrically, \(p_\theta\) trace a manifold within the probability simplex; projection (via KL) of \(p_{\text{data}}\) onto this manifold quantifies model capacity and misspecification.

## 7.2 Deconvolutional Latent Codes
Structured latents \(z=[z_{\text{type}}, z_{\text{pose}}]\) disentangle semantics (one-hot chair type) from pose (angles, distance). A deconvolutional generator upsamples these codes through transposed convolutions, BatchNorm, and ReLU blocks until producing RGB images. Training with paired \((x,z)\) minimizes \(\|x-G(z)\|_2^2\), and interpolation along \(z_{\text{type}}\) or pose parameters smoothly morphs category or viewpoint, demonstrating the learned manifold.

## 7.3 Latent Variable Perspective
Introducing latent causes \(z\) turns intractable \(p_{\text{data}}(x)\) into a joint \(p_\theta(x,z)=p(z)p_\theta(x|z)\) with simple prior (e.g. \(\mathcal{N}(0,I)\)) and decoder \(G_\theta(z)\). This generalizes factor analysis: nonlinear \(G_\theta\) “folds” the latent Gaussian into a multimodal data manifold, with \(z\) serving as coordinates on that manifold.

## 7.4 From Marginal to Joint KL and the ELBO
Augmenting data with latents gives \(p_{\text{data}}(x,z)=p_{\text{data}}(x)p_{\text{data}}(z|x)\). The joint KL decomposes as
$$
D_{\mathrm{KL}}(p_{\text{data}}(x,z)\Vert p_\theta(x,z))=D_{\mathrm{KL}}(p_{\text{data}}(x)\Vert p_\theta(x))+\mathbb{E}_{p_{\text{data}}(x)}D_{\mathrm{KL}}(p_{\text{data}}(z|x)\Vert p_\theta(z|x)).
$$
This yields the conceptual ELBO \(\mathbb{E}[\log p_\theta(x)]-\mathbb{E}D_{\mathrm{KL}}(p_{\text{data}}(z|x)\Vert p_\theta(z|x))\) and the computational ELBO \(\mathbb{E}_{p_{\text{data}}(z|x)}[\log p_\theta(x|z)]-D_{\mathrm{KL}}(p_{\text{data}}(z|x)\Vert p(z))\). The inference gap, model gap, and total joint KL correspond to the three stacked discrepancies illustrated in Fig. 7.3.

## 7.5 Learnable Inference and ELBO Insights
Replacing \(p_{\text{data}}(z|x)\) with a parametric \(q_\phi(z|x)\) leads to the VAE objective \(\min_{\theta,\phi}D_{\mathrm{KL}}(p_{\text{data}}(x)q_\phi(z|x)\Vert p(z)p_\theta(x|z))\) or equivalently \(\max_{\theta,\phi}\text{ELBO}\). Form 1, \(\log p_\theta(x)-D_{\mathrm{KL}}(q_\phi\Vert p_\theta(z|x))\), shows the ELBO as a likelihood lower bound tightened when inference matches the true posterior. Form 2 splits reconstruction and regularization: \(\mathbb{E}_{q_\phi}\log p_\theta(x|z)-D_{\mathrm{KL}}(q_\phi\Vert p(z))\). The asymmetry of KL induces mode-covering behavior when fitting \(p_\theta(x)\) and mode-seeking behavior when fitting \(q_\phi(z|x)\). EM reappears as the special case \(q_\phi=p_{\theta_t}(\cdot|x)\), yielding monotonic improvement via tight bounds per iteration; VAEs relax this by jointly learning \(\theta,\phi\).

## 7.6 VAE Implementation Details
Encoders output Gaussian parameters \(\mu_\phi(x), \sigma_\phi(x)\); decoders output Gaussian or Bernoulli likelihoods; priors remain \(\mathcal{N}(0,I)\). The reparameterization trick samples \(z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon\) with \(\varepsilon\sim\mathcal{N}(0,I)\), enabling backprop through stochastic nodes. The ELBO becomes an expectation of \(\log p_\theta(x|z)\) plus the analytic Gaussian KL \(\frac{1}{2}\sum_j (1+\log\sigma_{\phi,j}^2-\mu_{\phi,j}^2-\sigma_{\phi,j}^2)\). Training alternates sampling minibatches, drawing \(\varepsilon\), computing \(z\), evaluating the ELBO, and updating \(\theta,\phi\) with gradient ascent. Practical heuristics include KL annealing (\(\beta\)-VAE), rescaling reconstruction losses, and averaging over multiple \(\varepsilon\). Post-training, generation samples \(z\sim\mathcal{N}(0,I)\); reconstructions pass data through encoder then decoder; latent interpolation supports smooth traversals.

## 7.7 VAEs vs Diffusion Models
Diffusion models can be cast as VAEs whose latents are the entire noisy trajectory \((x_1,\dots,x_T)\). The forward process \(q(x_t|x_{t-1})\) is fixed Gaussian augmentation, eliminating the need to learn \(q_\phi\). This yields a supervised learning problem for the reverse kernels \(p_\theta(x_{t-1}|x_t)\), tighter ELBOs (since \(x_T\) is provably Gaussian and transitions are near-Gaussian), and more straightforward optimization. Thus diffusion succeeds by engineering the latent process so that inference is known rather than learned.

## 7.8 GANs and Wasserstein GAN
GANs frame generation as a binary classification game. Real data \(x\) carry label 1; generated samples \(G(z), z\sim\mathcal{N}(0,I)\) carry label 0. The discriminator maximizes \(\mathbb{E}_{p_{\text{data}}}\log D(x)+\mathbb{E}_{p_G}\log(1-D(x))\) while the generator minimizes it. In practice, \(-\log(1-D(G(z)))\) gives vanishing gradients early on, so \(\log D(G(z))\) is maximized instead. Wasserstein GAN replaces the sigmoid discriminator with a 1-Lipschitz critic \(f\) and optimizes the Earth-Mover distance \(\min_G \max_{f\in\mathcal{F}_L} \mathbb{E}_{p_{\text{data}}}f(x)-\mathbb{E}_z f(G(z))\), yielding continuous quality scores and more stable gradients. The chapter closes by analyzing mode collapse—when \(p_G\) covers only a few dominant modes—and summarizing mitigation strategies such as minibatch discrimination, unrolled updates, multiple critics, and objectives that explicitly penalize low diversity.

---
Chapter 7 therefore links maximum-likelihood geometry, ELBO derivations, and EM intuition to practical VAE training, contrasts learned inference with diffusion’s fixed forward process, and details the adversarial framing that powers GAN and Wasserstein GAN training.
