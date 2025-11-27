# Chapter 3 – Convolutional Neural Networks

## 3.0 Framing
CNNs are presented as computer programs built in the neural language of vectors and learnable transformations. They enforce the assumptions that local patterns matter, detectors can be reused at every spatial location, and higher layers should integrate information over progressively larger receptive fields. The result is an architecture that converts spatially arranged pixel vectors into a single high-level embedding by interleaving convolution, subsampling, and fully connected (FC) layers, as sketched in Figure 3.2 of the text.

## 3.1 Neural Networks as Computer Programs
The "neural language" is defined by the canonical layer update
\[
  h^{(l)} = \sigma\big(W^{(l)} h^{(l-1)} + b^{(l)}\big) \tag{3.1}
\]
so a neural network is a pipeline of thought vectors \(h^{(0)} \to h^{(1)} \to \cdots \to h^{(L)}\) \((3.2)\) that exchange information via linear maps, element-wise gates, and vector products. Programming with vectors means (i) storing distributed patterns in embeddings, (ii) composing them with the primitive operators above, and (iii) controlling dataflow via forward propagation and gradients supplied by backpropagation. Learning becomes "program writing": data labels define the loss, gradients set the edits, and gradient descent accumulates reusable subroutines. Understanding a trained network reduces to tracing what each vector encodes, how transformations remix components, and how representations split or merge across layers.

## 3.2 Computer Vision with CNNs
### 3.2.1 Input Structure and Representational Hierarchy
Each RGB pixel \(x_{ij} = (R_{ij}, G_{ij}, B_{ij})^\top\) forms the base vector \(h^{(0)}_{ij} = x_{ij}\) \((3.3)-(3.4)\). As depth increases, the spatial extent of the local vector \(h^{(l)}_{ij}\) grows and captures more abstract semantics: early layers detect 3×3 edges and color changes \((3.5)\), middle layers compose 7×7 or 11×11 object parts \((3.7)-(3.8)\), and late convolutional stacks aggregate full facial or car regions \((3.9)-(3.10)\). FC layers finally compress all spatial positions into global vectors for object configuration \(h^{(7)}\) and class logits \(h^{(8)}\) \((3.11)-(3.12)\). This pipeline yields the progression
\[
  \text{edges}_{3\times3} \;\Rightarrow\; \text{parts}_{7\times7} \;\Rightarrow\; \text{regions}_{15\times15} \;\Rightarrow\; \text{global objects} \tag{3.13}
\]
with increasing receptive field, semantic abstraction, and translation invariance.

### 3.2.2 Convolutional Computation and Dimensionality
A convolutional block centered at \((i,j)\) performs
\[
  h^{(l)}_{ij} = \sigma\!\left( \sum_{\Delta i=-r}^r \sum_{\Delta j=-r}^r W_{\Delta i, \Delta j}\, h^{(l-1)}_{i+\Delta i,\, j+\Delta j} + b^{(l)} \right) \tag{3.14}
\]
which is equivalent to concatenating the \((2r+1)^2\) local vectors and applying one large linear map \((3.15)\). To form richer compositions we typically expand the channel dimension, enforcing \(d_l > d_{l-1}\) \((3.16)\) so more filters can summarize the larger neighborhoods. Fully connected readouts use
\[
  h^{(l)} = \sigma\!\left( \sum_{i,j} W_{ij} h^{(l-1)}_{ij} + b^{(l)} \right) \tag{3.29}
\]
turning the spatial grid into a single global thought vector.

### 3.2.3 Inductive Bias and Subsampling
The convolutional form \(h^{(l)}_{ij} = f\big(\sum W_{\Delta i, \Delta j} h^{(l-1)}_{i+\Delta i, j+\Delta j}\big)\) \((3.17)\) hard-codes three inductive biases:
1. **Translation invariance**: identical kernels \(W_{ij} = W\) across space \((3.18)\).
2. **Local connectivity**: weights vanish for \(|\Delta i|, |\Delta j| > k\) \((3.19)\), assuming distant pixels contribute weakly.
3. **Hierarchical composition**: stacking layers enlarges the receptive field \((3.20)\).
These biases dramatically reduce parameters \((3.21)\), data needs \((3.22)\), and expected test error \((3.23)\) on natural images, at the cost of flexibility on non-visual data. Strided subsampling keeps only \(h^{(l)}_{si,sj}\) \((3.24)-(3.28)\), halving resolution while doubling effective receptive field per layer, so later detectors reason over broader context with manageable compute.

### 3.2.4 Channel/Kernel Views and 1×1 Convolutions
Explicitly tracking channels yields
\[
  h^{(l)}_{ijk} = \sigma\!\left( \sum_{c=1}^{d_{l-1}} \sum_{\Delta i,\Delta j} W_{k,c,\Delta i, \Delta j} \, h^{(l-1)}_{i+\Delta i, j+\Delta j, c} + b^{(l)}_k \right) \tag{3.30}
\]
so each output channel \(k\) aggregates every input channel \(c\) over the local neighborhood. Setting \(\Delta i = \Delta j = 0\) produces a 1×1 convolution
\[
  h^{(l)}_{ij} = \sigma\!\left(W_0 h^{(l-1)}_{ij} + b^{(l)}\right) \tag{3.31}
\]
which behaves like a position-wise MLP that mixes channels without expanding the receptive field. Such layers efficiently compress or expand dimensions \((3.32)-(3.33)\), enable cross-channel interactions \((3.34)\), and parallelize well because each spatial location is independent.

## 3.3 Backpropagation in CNNs
The chapter derives gradients starting from the log-likelihood objective \(J = \log p(y\mid s)\) with scores \(s = h^{(L)} = W^{(L)} h^{(L-1)}\) \((3.35)\). The error signal at the head is the usual \(\partial J/\partial s = y - p\) \((3.36)\). FC layers backprop exactly as in MLPs:
\[
  h^{(l)} = W^{(l)} h^{(l-1)}, \quad \frac{\partial J}{\partial h^{(l-1)}} = (W^{(l)})^\top \frac{\partial J}{\partial h^{(l)}}, \quad \frac{\partial J}{\partial W^{(l)}} = \frac{\partial J}{\partial h^{(l)}} (h^{(l-1)})^\top.
\]
These correspond to Equations (3.37)–(3.39). For convolutional layers the pre-activations and activations follow
\[
  s^{(l)}_{ij} = \sum_{\Delta i,\Delta j} W^{(l)}_{\Delta i,\Delta j} h^{(l-1)}_{i+\Delta i, j+\Delta j}, \quad h^{(l)}_{ij} = \sigma\big(s^{(l)}_{ij}\big),
\]
matching Equations (3.40)–(3.41) and leading to
\[
  \frac{\partial J}{\partial s^{(l)}_{ij}} = \frac{\partial J}{\partial h^{(l)}_{ij}} \odot \sigma'\big(s^{(l)}_{ij}\big),
\]
\[
  \frac{\partial J}{\partial h^{(l-1)}_{ij}} = \sum_{\Delta i,\Delta j} (W^{(l)}_{\Delta i,\Delta j})^\top \frac{\partial J}{\partial s^{(l)}_{i-\Delta i, j-\Delta j}},
\]
\[
  \frac{\partial J}{\partial W^{(l)}_{\Delta i,\Delta j}} = \sum_{i,j} \frac{\partial J}{\partial s^{(l)}_{ij}} (h^{(l-1)}_{i+\Delta i, j+\Delta j})^\top,
\]
which recover Equations (3.42)–(3.44). When a layer subsamples with stride \(s\), each upstream gradient \(\partial J/\partial h^{(l)}_{si,sj}\) is copied to the appropriate downsampled position \((3.45)-(3.46)\). Implementation-wise, CNNs expose parallel work at three levels—independent spatial positions, feature channels, and batch items \((3.47)-(3.49)\)—which maps naturally to GPU/TPU hardware for both forward and backward passes.

---
This summary highlights how Chapter 3 recasts convolutional networks as vector programs whose structure encodes translation invariance, local composition, and efficient gradient computation, enabling scalable visual recognition pipelines.
