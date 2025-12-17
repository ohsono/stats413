# Chapter 5 – Transformer and GPT

## 5.0 Framing
Transformers are presented as neural programs that maintain a residual stream where high-dimensional "thought vectors" accumulate nearly orthogonal features. Each layer alternates between two retrieval modes—attention for contextual lookups and MLPs (optionally augmented with experts) for associative recall—so the stream fuses document-specific cues with world knowledge. The chapter then follows the architecture from embeddings through GPT-style decoding, scaling trends, and the modern instruction-tuning/RLHF pipeline, before surveying vision/multimodal variants such as ViT and CLIP.

## 5.1 Distributed Representations and Residual Assembly Lines
A token such as “Barack Obama” is embedded into $h = \text{NN}(x) \in \mathbb{R}^d$ $(5.1)$, which decomposes as a superposition $h = g_1 + g_2 + \dots$ with near-orthogonal components $(5.2)-(5.3)$ capturing party, office, biography, etc. Projection heads extract aspects via matrices like $g_{\text{party}} = W_{\text{party}} h$ $(5.4)-(5.9)$. The residual stream iteratively adds features—$h^{(1)} = W_{\text{embed}} x$, $h^{(l+1)} = h^{(l)} + g^{(l)}$ $(5.10)-(5.12)$—so each layer behaves like an assembly-line station enriching the superposed representation $(5.13)-(5.17)$.

## 5.2 Transformer Residual Stream and Retrieval Modes
Each layer reads the residual state, performs attention to retrieve context and an MLP to retrieve memorized associations, then writes the enriched vector back $(5.18)-(5.24)$. Context retrieval uses attention weights $\alpha_{ti} = \operatorname{softmax}(q_t^\top k_i / \sqrt{d_k})$ with query/key/value projections $(5.27)-(5.36)$, yielding $h_{t}^{(l+1/2)} = h_t^{(l)} + \sum_i \alpha_{ti} v_i$ $(5.33)-(5.34)$. MLPs (optionally organized as mixtures of experts with router weights $g_t$ $(5.37)-(5.42)$) then inject stored knowledge. Multi-head attention provides parallel context views, while MoE layers supply specialized associative recall.

## 5.3 Full Transformer Stack
Tokens receive learned or sinusoidal positional encodings $e_t = W_{\text{embed}} x_t + W_{\text{pos}} t$ $(5.43)-(5.47)$. Each block applies causal attention and an MLP through the residual stream $(5.48)-(5.54)$, and the final state is unembedded for next-token logits $(5.55)-(5.57)$. Backpropagation remains highly parallel because attention matrix products and per-position MLPs can be batched $(5.58)-(5.62)$, enabling deep stacks across devices.

## 5.4 Associative Memory View
Linear layers store query–answer pairs through their SVD $W = \sum_i \lambda_i b_i a_i^\top$ $(5.63)-(5.70)$, so feeding $x = \sum_i c_i a_i$ produces $y = \sum_i c_i b_i$. Rank-1 edits $(5.80)-(5.84)$ or low-rank LoRA adapters $W_{\text{new}} = W_{\text{pre}} + BA$ $(5.85)-(5.90)$ add new memories without disturbing the base model. Even QKV projections $(5.91)-(5.93)$ can be read as associative lookups that prime the attention mechanism.

## 5.5 Matrix Reasoning vs Symbolic Logic
Matrix transforms implement thousands of implicit rules via linear combinations $(5.94)-(5.103)$ and learnable gradients $(5.104)-(5.108)$, composing seamlessly across layers $(5.109)-(5.113)$. Yet symbolic logic retains advantages for universally quantified statements $(5.114)-(5.117)$, motivating hybrid neuro-symbolic systems.

## 5.6 Architectural Comparison
Temporal convolutions and GPT both access the entire prefix $(5.118)-(5.119)$, whereas RNNs compress history into $h_t = f(h_{t-1}, x_t)$ $(5.120)-(5.121)$. GPT’s key/value cache $(5.122)-(5.124)$ acts like persistent notes, trading higher memory for perfect recall, while RNNs rely on limited working memory. This explains GPT’s superior handling of long-range dependencies despite increased compute.

## 5.7 Encoder–Decoder Transformer and 5.8 BERT vs GPT
The original translation model encodes the source via bidirectional attention and decodes with causal self-attention plus cross-attention $(5.125)-(5.130)$. BERT keeps the encoder stack, masking tokens for bidirectional understanding, whereas GPT keeps only the causal decoder, making it ideal for autoregressive generation without special classification tokens.

## 5.9 Transformer Reference Sizes and 5.10 GPT‑3
The 2017 Transformer used $d_{\text{model}} = 512$ (base) or 1024 (big) with $d_{\text{ff}} = 4 d_{\text{model}}$, $h = 8/16$ heads, and six encoder/decoder layers $(5.137)-(5.145)$. GPT‑3 expands to $d = 12{,}288$, $L = 96$, $H = 96$ with per-layer parameter counts $P_{\text{attn}} = 4 d^2$ and $P_{\text{MLP}} = 8 d^2$ $(5.146)-(5.157)$, highlighting the compute/memory footprint of modern LLMs.

## 5.11 Scaling Laws
Empirical power laws link loss $L$ to parameters $N$, data $D$, and compute $C$: $L = (N_c/N)^{\alpha_N} + c_1$, $L = (D_c/D)^{\alpha_D} + c_2$, $L = (C_c/C)^{\alpha_C} + c_3$ $(5.158)-(5.160)$. For a compute budget $C$, optimal allocations follow $N \propto C^{0.6}, D \propto C^{0.4}$ $(5.161)-(5.162)$; Chinchilla scaling argues for $N \propto D \propto C^{0.5}$ $(5.163)-(5.164)$, implying many released LLMs are undertrained on data.

## 5.12 Two-Stage Training
LLMs first undergo unsupervised pre-training via next-token prediction $(5.169)-(5.172)$, acquiring linguistic and world knowledge. Instruction tuning reformats tasks as natural-language prompts with desired outputs and continues autoregressive training on curated pairs $(5.173)$, enabling few-shot behaviors and better alignment.

## 5.14 Reinforcement Learning from Human Feedback (RLHF)
A reward model shares the base architecture but ends with a scalar head $r_\phi(x,y) = w^\top h_{\text{final}}$ $(5.174)-(5.175)$. Trained through Bradley–Terry comparisons $(5.176)-(5.183)$, it scores candidate responses. Policy gradients adjust the language model to maximize expected reward $(5.184)-(5.189)$, with baselines/value functions $(5.190)-(5.197)$ reducing variance. Unlike maximum-likelihood imitation $(5.199)-(5.203)$, RLHF optimizes self-generated outputs. Proximal Policy Optimization rewrites the objective using importance ratios $\rho_\theta$ $(5.204)-(5.207)$ and clamps updates via $\min(\rho A, \operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A)$ $(5.208)-(5.209)$, stabilizing alignment training.

## 5.16 Vision Transformer (ViT)
Images are split into $N = HW/P^2$ patches $(5.211)-(5.213)$, linearly embedded with positional terms $(5.214)-(5.215)$, and processed by standard transformer blocks $(5.216)-(5.217)$. ViT offers global receptive fields from the first layer at the cost of quadratic attention, requiring large pre-training datasets compared with CNNs.

## 5.17 CLIP
CLIP trains dual encoders $z_{\text{img}} = f_{\text{img}}(\text{image})$, $z_{\text{text}} = f_{\text{text}}(\text{text})$ $(5.218)-(5.220)$ and optimizes a symmetric contrastive loss $(5.221)-(5.224)$ over batchwise similarity matrices. A temperature hyperparameter $\tau$ tunes the sharpness, yielding aligned vision-language representations that support zero-shot classification, retrieval, and semantic search.

---
Chapter 5 thus frames Transformers as residual assembly lines combining attention-based context retrieval with associative MLP memory, scales them from the original encoder–decoder to GPT‑3, and details the modern training stack (pre-training → instruction tuning → RLHF) alongside vision and multimodal adaptations.
