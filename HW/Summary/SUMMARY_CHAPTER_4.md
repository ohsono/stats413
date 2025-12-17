# Chapter 4 – Recurrent Neural Networks

## 4.0 Framing
Chapter 4 develops RNNs as vector programs that evolve through real time (memory streams) and computational time (residual streams). Starting from the vanilla tanh recurrence used for next-word prediction, it diagnoses vanishing gradients, then motivates LSTM-style memory streams and residual connections as dual assembly lines that preserve information horizontally across sequence length and vertically across depth. The latter half of the chapter unifies RNNs with temporal CNNs via state space models, extends the view to continuous time and selective (Mamba) architectures, and closes with an RNN interpretation of quantum mechanics.

## 4.1 Vector Evolution Over Time
At step \(t\) the core update
$$
  h_t = \tanh\!\left(W_{\text{recurrent}} h_{t-1} + W_{\text{embed}} x_t + b\right) \tag{4.1}
$$
keeps the neural language intact while indexing vectors by time. The hyperbolic tangent \(\tanh(x) = (e^{2x}-1)/(e^{2x}+1)\) \((4.2)\) supplies a zero-centered range \((-1,1)\) and strong gradients near the origin, letting the hidden vector act as a rolling memory that accumulates semantic and syntactic information.

## 4.2 Next-Word Prediction and BPTT
Forward computation chains three tokens \(x_1,x_2,x_3\) into \(h_1,h_2,h_3\) \((4.3)-(4.6)\), then scores the next word via \(s = W_{\text{unembed}} h_3\) and \(p = \text{softmax}(s)\) \((4.7)-(4.8)\). Backpropagation through time begins with \(\partial J/\partial s = y - p\) \((4.9)\) and propagates through the unembedding \((4.10)-(4.11)\) before marching backward in time using
$$
  \frac{\partial J}{\partial h_{t-1}} = W_{\text{recurrent}}^\top \left(\frac{\partial J}{\partial h_t} \odot (1 - h_t^2)\right),\tag{4.12}
$$
with accumulated gradients for \(W_{\text{recurrent}}\) and \(W_{\text{embed}}\) \((4.13)-(4.14)\). The signal degrades geometrically as
$$
  \left\|\frac{\partial J}{\partial h_3}\right\| \approx c,\quad \left\|\frac{\partial J}{\partial h_2}\right\| \approx c\alpha,\quad \left\|\frac{\partial J}{\partial h_1}\right\| \approx c\alpha^2 \tag{4.19)-(4.21}
$$
with \(0<\alpha<1\), reflecting the bounded derivative of tanh \((4.16)-(4.18)\) and repeated multiplication by \(W_{\text{recurrent}}\) \((4.15)\). Eigenvalues \(|\lambda|<1\) drive vanishing gradients, while \(|\lambda|>1\) risks explosion, complicating long-range credit assignment.

## 4.3 Memory Stream Innovation
Rewriting the recurrence with an additive skip connection
$$
  h_t = h_{t-1} + \tanh\!\left(W_{\text{recurrent}} h_{t-1} + W_{\text{embed}} x_t + b\right) \tag{4.22)-(4.23}
$$
reveals a **memory stream**: an assembly line where the previous state passes through unchanged and new updates accumulate. Because
$$
  \frac{\partial h_t}{\partial h_{t-1}} = I + W_{\text{recurrent}}^\top \operatorname{diag}\big(1 - \tanh^2(\cdot)\big) \tag{4.27}
$$
the identity term furnishes a direct gradient path (Path 1) in addition to the three multiplicative paths through update branches \((4.39)-(4.43)\). High-dimensional states can store superposed updates \(h_3 = h_0 + \Delta h_1 + \Delta h_2 + \Delta h_3\) \((4.33)-(4.38)\) as long as the increments are near-orthogonal, allowing subject, action, and object semantics to coexist and remain retrievable. LSTMs formalize the division between long-term memory in weights, short-term memory in the cell stream \(c_t\), and working memory in \(h_t\) \((4.44)-(4.50)).

## 4.4 LSTM Gating Mechanism
LSTMs add gates to the memory stream:
$$
  f_t = \sigma(W_f [h_{t-1}, x_t] + b_f),\quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i),\quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \tag{4.45)-(4.47}
$$
The candidate update \(\Delta c_t = \tanh(W_c [h_{t-1}, x_t] + b_c)\) \((4.48)\) is fused via
$$
  c_t = c_{t-1} \odot f_t + \Delta c_t \odot i_t, \qquad h_t = \tanh(c_t) \odot o_t \tag{4.49)-(4.50}
$$
so that the cell stream maintains additive persistence while the learned gates decide what to keep, add, or expose. This makes short-term memory both longer-lived and selectively updated. GRUs emerge as a simplified variant that merges \(c_t\) and \(h_t\) while retaining the essential memory-stream idea.

## 4.5 Multi-layer Recurrent Networks
Stacking layers yields
$$
  h^{(1)}_t = \tanh(W^{(1)}_{\text{recurrent}} h^{(1)}_{t-1} + W^{(1)}_{\text{embed}} x_t + b^{(1)}),
$$
$$
  h^{(2)}_t = \tanh(W^{(2)}_{\text{recurrent}} h^{(2)}_{t-1} + W^{(2)}_{\text{associative}} h^{(1)}_t + b^{(2)}) \tag{4.51)-(4.52}
$$
with logits \(s = W_{\text{unembed}} h^{(2)}_t\) \((4.53)-(4.54)\). Backpropagation must traverse depth and time sequentially, producing gradients such as \(\partial J/\partial h^{(2)}_3\) \((4.55)\) before distributing signals to lower layers \((4.56)-(4.60)\). Training therefore stores all intermediate states and suffers compounded vanishing through both dimensions. Inference, by contrast, needs only the current hidden states \((4.61)-(4.64)\). Adding memory streams per layer \((4.65)-(4.70)\) lets each depth level maintain its own additive record, promoting stable gradient flow horizontally.

## 4.6 Residual Stream Through Depth
Residual connections reapply the assembly-line idea vertically:
$$
  h^{(2)}_t = h^{(1)}_t + \tanh\!\left(W^{(2)}_{\text{recurrent}} h^{(2)}_{t-1} + W^{(2)}_{\text{associative}} h^{(1)}_t + b^{(2)}\right) \tag{4.71)-(4.72}
$$
so
$$
  \frac{\partial h^{(2)}_t}{\partial h^{(1)}_t} = I + W^{(2)\top}_{\text{associative}} \operatorname{diag}\big(1 - \tanh^2(\cdot)\big) \tag{4.73}
$$
provides a direct vertical gradient path. The same idea underlies residual MLPs: treat layer index \(l\) as computational time \(t\), initialize \(h_0 = W_{\text{embed}} x\) \((4.74)-(4.78)\), and iterate
$$
  h_t = h_{t-1} + \sigma(W^{(t)}_{\text{residual}} h_{t-1} + b^{(t)}) \tag{4.79}
$$
until \(s = W_{\text{unembed}} h_T\) \((4.80)\). The residual stream thus supplies an assembly line in computational time, enabling successive refinements while keeping gradients intact \((4.81)-(4.89)\).

## 4.7 Residual Stream as Learned Iterative Algorithm
The recurrence \(h_t = h_{t-1} + \sigma(W^{(t)}_{\text{residual}} h_{t-1} + b^{(t)})\) \((4.90)\) mirrors finite-step iterative methods like gradient ascent \(x_t = x_{t-1} + \eta \nabla f(x_{t-1})\) \((4.91)\), yet it learns both the “gradient” and “step size” directly from data via the update \(\Delta h_t\) \((4.92)\). Because the loop runs for a fixed \(T\) steps \((4.94)-(4.97)\), each iteration must perform a meaningful refinement, effectively learning a task-specific optimization procedure without an explicit objective.

## 4.8 Neural Programming Language
Residual and memory streams show how data writes neural programs: vectors are the data structures, matrix multiplies and nonlinearities are the primitive instructions, and backpropagation tunes parameters so that multi-step residual loops act like for-loops \((4.93)-(4.97)\). This simple “neural language” becomes the foundation for digital intelligence because it enables deep computation with reliable gradient flow.

## 4.9 Parameter Sharing Across Streams
Temporal memory streams usually share \(W_{\text{recurrent}}\) across time \((4.98)\) to reflect stationary sequence dynamics, whereas computational residual streams may use step-specific weights \(W^{(t)}_{\text{residual}}\) \((4.99)\) so that early refinements differ from later ones. The chapter also shows how to weave residual micro-iterations into CNN layers: initialize \(h^{(l,0)}\) with a standard convolution \((4.101)\) and run \(T_l\) refinement steps
$$
  h^{(l,t)}_{ij} = h^{(l,t-1)}_{ij} + \sigma\!\left(\sum_{\Delta i,\Delta j} W^{(l,t)}_{\Delta i,\Delta j} h^{(l,t-1)}_{i+\Delta i, j+\Delta j} + b^{(l,t)}\right) \tag{4.102}
$$
so each layer learns both initial detectors and multi-step refinements.

## 4.10 Vanilla RNNs vs Temporal CNNs
Standard RNNs rely on horizontal recurrence \((4.103)-(4.104)\) and must be trained sequentially via BPTT, but they support arbitrary sequence length at inference. Temporal CNNs replace horizontal recurrence with causal convolutions over past embeddings \((4.105)-(4.106)\), enabling fully parallel training but imposing a fixed receptive field. State space models reconcile these extremes.

## 4.11 State Space Models (SSMs)
A linear SSM
$$
  h_t = A h_{t-1} + B x_t, \qquad y_t = C h_t \tag{4.107)-(4.108}
$$
expands to a convolution \(y_t = \sum_{\Delta t \ge 0} W_{\Delta t} x_{t-\Delta t}\) with \(W_{\Delta t} = C A^{\Delta t} B\) \((4.109)-(4.116)\), showing that recurrent and convolutional views are the same factorization. Sequential inference uses the recurrence, while parallel training uses the convolution, yielding parameter efficiency through the shared matrices \(A,B,C\).

## 4.12 Continuous-Time SSMs
Starting from \(h'(t) = A h(t) + B x(t)\) \((4.117)\), discretization over step \(\Delta\) gives the memory-stream form \(h(t+\Delta) = (I + A\Delta/N)^N h(t)\) \((4.118)-(4.126)\) and, in the limit, \(A_d = e^{A\Delta}\), \(B_d = (e^{A\Delta}-I) A^{-1} B\) \((4.137)-(4.139)\). The discrete update is \(h(t+\Delta) = A_d h(t) + B_d x(t)\) \((4.140)\), clarifying how continuous dynamics map onto residual-style discrete computations.

## 4.13 Mamba: Selective State Space Model
Mamba augments SSMs with input-dependent parameters
$$
  \Delta_t, A_t, B_t, C_t = \text{MLPs}(x_t), \qquad h_t = e^{A_t \Delta_t} h_{t-1} + B_t x_t, \qquad y_t = C_t h_t \tag{4.141)-(4.143}
$$
so the model can adapt its time step, dynamics, and readout based on the current token. This preserves the efficiency of linear recurrences for inference, retains convolutional parallelism for training, and injects Transformer-like selectivity via the conditioning networks.

## 4.14 Quantum Mechanics as an RNN
Quantum dynamics are recast as a fixed RNN where measurements play the role of inputs: observations are embedded as \(h(0) = W_{\text{embed}} x(0)\) \((4.144)\), evolve via the Schrödinger equation \(h'(t) = -i H h(t)\) (so \(W_{\text{recurrent}} = -iH\)) \((4.145)-(4.150)\), and are projected back through \(s(t) = W_{\text{unembed}} h(t)\) before applying a squared-softmax \(p_i(t) = |s_i(t)|^2 / \|s(t)\|_2^2\) \((4.146)-(4.154)\). Measurements collapse the state via the same embedding operator \((4.149)-(4.158)\), turning the hidden “game engine” of quantum reality into a classical “rendered display.” The observer’s choice of basis \(E\) thus defines the interface between the continuous unitary evolution and the discrete measurement outcomes.

---
Chapter 4 portrays RNNs as neural programs equipped with two orthogonal highways—memory streams through time and residual streams through depth—that protect gradient flow, enabling long-horizon reasoning, multi-step computation, and unifying perspectives ranging from language modeling to quantum physics.
