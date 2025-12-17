# Chapter 2 – Multi-Layer Perceptron

## 2.0 Framing
The chapter transitions from classical statistical models to neural networks by showing how logistic regression naturally becomes the front-end of a perceptron. Embeddings, associative memory, recommender systems, normalization, and dropout are introduced as higher-level constructs built atop this foundation.

## 2.1 Logistic Regression as a Perceptron Layer
The familiar score function
  $$s_i = x_i^\top \beta + \beta_0 = x_i^\top w + b$$
pairs with the sigmoid probability
$$
  p_i = \sigma(s_i) = \frac{1}{1 + e^{-s_i}}.
$$
Mapping the coefficients to weights and the intercept to a bias yields the simplest perceptron, showing that neural notation is simply a relabeling of GLM components. The architecture figure reinterprets each input feature as a node, weights as edges, and the sigmoid as the activation of the output neuron.

## 2.2 Single Hidden Layer Networks
A one-hidden-layer MLP with ReLU units is described by
$$
  h_k = \text{ReLU}(w_k^{(1)} x + b_k^{(1)}) = \max(0, w_k^{(1)} x + b_k^{(1)}), \quad
  s = b^{(2)} + \sum_k w_k^{(2)} h_k.
$$
With multi-dimensional inputs each hidden unit creates a fold along the hyperplane \(w_{k1}^{(1)} x_1 + w_{k2}^{(1)} x_2 + b_k^{(1)} = 0\), turning the network into a piecewise-linear interpolant. Likelihood objectives produce error signals \(e = y - s\) (regression) or \(e = y - p\) (classification) that backpropagate through the ReLU gates:
$$
  \frac{\partial J}{\partial w_k^{(2)}} = e h_k,\qquad
  \delta_k = \frac{\partial J}{\partial h_k} = e w_k^{(2)},\qquad
  \frac{\partial J}{\partial w_k^{(1)}} = \delta_k \cdot \text{ReLU}'(w_k^{(1)} x + b_k^{(1)}) x.
$$
When the number of hidden units \(d\) is large enough—e.g. \((d_{in}+1)d + (d+1) > n\)—the model enters an overparameterized regime capable of perfect interpolation. Gradient descent from zero initialization acts as an implicit regularizer that favors minimum-norm solutions and smooth interpolants despite fitting the training data exactly.

## 2.3 General MLPs and Backpropagation
Stacking layers yields the generic form
$$
  s^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \qquad h^{(l)} = \sigma(s^{(l)}), \qquad s = W^{(L)} h^{(L-1)} + b^{(L)}.
$$
Backpropagation applies the chain rule recursively:
$$
  \frac{\partial J}{\partial s^{(l)}} = \frac{\partial J}{\partial h^{(l)}} \odot \sigma'(s^{(l)}), \qquad
  \frac{\partial J}{\partial h^{(l-1)}} = (W^{(l)})^\top \frac{\partial J}{\partial s^{(l)}},
$$
$$
  \frac{\partial J}{\partial W^{(l)}} = \frac{\partial J}{\partial s^{(l)}} (h^{(l-1)})^\top, \qquad
  \frac{\partial J}{\partial b^{(l)}} = \frac{\partial J}{\partial s^{(l)}}.
$$
Component-wise derivations, along with both row-vector and column-vector interpretations, emphasize why gradients can be computed efficiently, enabling deep architectures.

## 2.5 Optimization, Initialization, and Adaptive Methods
Mini-batch SGD partitions the dataset into batches \(B_t\) and uses the update
$$
  g_t = -\frac{1}{B} \sum_{i\in B_t} \nabla_\theta J_i(\theta), \qquad
  \theta_{t+1} = \theta_t - \eta g_t,
$$
with variance \(\text{Var}[g_t] \propto 1/B\). Adam augments SGD with exponentially weighted moments:
$$
  v_t = \beta_1 v_{t-1} + g_t, \quad G_t = \beta_2 G_{t-1} + g_t^2, \quad
  \theta_t = \theta_{t-1} - \eta \frac{v_t}{\sqrt{G_t + \epsilon}}.
$$
Initialization strategies compare zero, Gaussian, Xavier \(\mathcal{N}(0, \frac{2}{n_{in}+n_{out}})\), and He \(\mathcal{N}(0, \frac{2}{n_{in}})\) schemes, highlighting their impact on variance preservation, symmetry breaking, and gradient flow.

## 2.8 Multi-class Classification and Representation Collapse
For labels \(y \in \{0,1\}^C\) the softmax
$$
  p_c = \text{softmax}(s)_c = \frac{e^{s_c}}{\sum_j e^{s_j}}
$$
leads to log-likelihood
$$
  J = \sum_{c=1}^C y_c s_c - \log\Big(\sum_{j=1}^C e^{s_j}\Big), \qquad \frac{\partial J}{\partial s} = y - p.
$$
Binary sigmoid is recovered by fixing one score to zero. Deeper layers exhibit hierarchical abstraction, progressively reducing nuisance variation so that \(\text{dim}~\text{Var}(h^{(l)}|y)\) shrinks even as mutual information with the label \(I(h^{(l)}; y)\) grows, yielding within-class convergence and between-class separation.

## 2.9 Word Embedding
For next-word prediction, the model uses
$$
  h = W_{\text{embed}} x, \qquad s = W_{\text{unembed}} h, \qquad p = \text{softmax}(s),
$$
where \(x\) and \(y\) are one-hot vectors. Gradients are
$$
  \frac{\partial J}{\partial W_{\text{unembed}}} = (y - p) h^\top,
  \quad \frac{\partial J}{\partial h} = (y - p)^\top W_{\text{unembed}},
$$
$$
  \frac{\partial J}{\partial W_{\text{embed}}} = \Big((y - p)^\top W_{\text{unembed}}\Big) x^\top,
$$
updating only the embedding column corresponding to the active word. Embeddings act as dense "thought vectors" that transform sparse symbolic inputs into continuous representations supporting analogical reasoning (e.g., \(\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})\)).

## 2.10 Embedding Concepts
Embeddings facilitate linear transforms \(Wh\), element-wise operations \(\sigma(h)\), and vector arithmetic. Distances encode similarity \(\|\text{vec}(\text{cat}) - \text{vec}(\text{dog})\| < \|\text{vec}(\text{cat}) - \text{vec}(\text{car})\|\), while directions capture relations \(\text{vec}(\text{Paris}) - \text{vec}(\text{France}) \approx \text{vec}(\text{Berlin}) - \text{vec}(\text{Germany})\). Their learnability through large-scale optimization makes them central to modern deep learning by bridging discrete symbols and continuous computation.

## 2.11 Associative Memory
A tied-weight associative memory uses
$$
  h^{(1)} = W_{\text{embed}} x, \qquad h^{(2)} = W_{\text{assoc}} h^{(1)}, \qquad s = W_{\text{embed}}^\top h^{(2)},
$$
with gradients combining embedding and association terms. Linear associative memories define
$$
  W_{\text{assoc}} = \sum_{i=1}^n b_i a_i^\top,
$$
which yields exact recall \(W_{\text{assoc}} a_k = b_k\) when \(\{a_i\}\) are orthonormal. Non-linear variants insert hidden layers \(h^{(2)} = \sigma(W_{\text{assoc}}^{(\text{in})} h^{(1)})\), \(h^{(3)} = W_{\text{assoc}}^{(\text{out})} h^{(2)}\) to interpolate between memories.

## 2.12 Embeddings for Recommender Systems
Users and items receive embeddings \(a_i, b_j \in \mathbb{R}^d\), and ratings are predicted via
$$
  r_{ij} = a_i^\top b_j = \sum_{k=1}^d a_{ik} b_{jk}.
$$
Fitting minimizes
$$
  \min_{\{a_i, b_j\}} \sum_{(i,j) \in \Omega} \frac{1}{2} (r_{ij} - a_i^\top b_j)^2,
$$
which can be interpreted as an embedding lookup followed by a user-specific readout. Nonlinear mappings \(r_{ij} = f(a_i, b_j)\) are possible, but the dot-product provides interpretability and a clear mechanism for modeling preference "addiction." 

## 2.13 Superposition
Embeddings reside in orthogonal bases \(h = \sum_i c_i b_i\) or subspaces \(h = \sum_k B_k C_k\), meaning semantic concepts are distributed rather than tied to single coordinates. Example decompositions for "Barack Obama" include presidency, ethnicity, education, family, and political subspaces, which remain approximately orthogonal \(B_j^\top B_k \approx 0\) for \(j \ne k\). Neural networks exploit these subspaces through learned attention and non-linear interactions.

## 2.14 RMS Normalization
Root-mean-square normalization rescales vectors via
$$
  \text{RMSNorm}(h) = \frac{h}{\|h\|/\sqrt{d}},
$$
projecting them onto a sphere of radius \(\sqrt{d}\) and making cosine similarity an inherent measure: \(\langle \text{RMSNorm}(h_1), \text{RMSNorm}(h_2) \rangle = \cos \theta_{12}\). It yields scale invariance, smooth loss surfaces, bounded gradients \(\partial \text{RMSNorm}(h)/\partial h\), and practical layers of the form \(\text{out} = \text{RMSNorm}(h) \gamma\).

## 2.15 Dropout and Fault Tolerance
Dropout masks activations during training:
$$
  h_{\text{drop}} = m \odot h, \qquad P(m_i = 0) = p,
$$
with expectations
$$
  \mathbb{E}[h_{\text{drop}}] = (1-p) h, \qquad \text{Var}[h_{\text{drop}}] = p(1-p) (h \odot h).
$$
Testing rescales outputs via \(s = W_{\text{out}}^\top h (1-p)\) to match expected activations. Dropout behaves like averaging over \(2^d\) sub-networks, preventing feature co-adaptation and encouraging distributed representations. Compared with RMSNorm—which corrects magnitude errors through \(\text{RMSNorm}(\alpha h + \epsilon) \approx \text{RMSNorm}(h)\)—dropout handles structural faults, and using both yields guarantees such as \(\| \text{RMSNorm}(m \odot h) \| = \sqrt{d}\) and \( \mathbb{E}[m \odot \text{RMSNorm}(h)] = (1-p) \text{RMSNorm}(h)\).

---
This summary condenses the mathematical and conceptual highlights of Chapter 2, emphasizing how linear models evolve into deep multi-layer perceptrons equipped with embeddings, associative memory, and robust optimization techniques.
