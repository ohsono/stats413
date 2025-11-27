# Chapter 8 – Deep Reinforcement Learning

## 8.0 Framing
Chapter 8 treats AlphaGo/AlphaGo Zero and Atari DQN/PPO as foil domains to highlight fundamental RL dichotomies: model-based vs model-free, value vs policy, on-policy vs off-policy, dense vs sparse rewards, and policy/execution vs explicit planning. The narrative progresses from MDP theory and policy-gradient/Bellman foundations, through AlphaGo’s neural-MCTS pipeline, to Atari DQN and policy-gradient implementations, ending with reflections on planning (System 2) vs intuition (System 1).

## 8.1 MDP Primer
An MDP comprises states, actions, transitions \(P(s'|s,a)\), rewards \(R(s,a,s')\), and discount \(\gamma\). Policies \(\pi(a|s)\) induce value functions \(V_\pi\) and action-values \(Q_\pi\) (Eq. 8.1–8.4); optimal values satisfy \(V^*(s)=\max_a Q^*(s,a)\). The first dichotomy arises between model-based planning (e.g., dynamic programming, MCTS) and model-free learning (TD, Q-learning, policy gradient).

## 8.2 Fundamental Theorems and Algorithms
The policy-gradient theorem gives \(\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) Q_{\pi}(s,a)]\) (Eq. 8.7–8.12), enabling direct policy optimization. Value-based relationships yield Bellman equations \(V_\pi(s)=\sum_a \pi(a|s)[r+\gamma \sum_{s'}P(s'|s,a)V_\pi(s')]\) (Eq. 8.20) and the Bellman optimality operator \(TQ = r + \gamma \mathbb{E}_{s'} \max_{a'} Q(s',a')\) (Eq. 8.27), a contraction with unique fixed point \(Q^*\). These theorems underpin policy gradients, Q-learning updates \(Q \leftarrow Q + \alpha(r+\gamma\max_{a'}Q(s',a')-Q)\) (Eq. 8.31–8.32), MPC formulations, and actor-critic bootstrapping (Eq. 8.35–8.38).

## 8.3–8.9 AlphaGo/AlphaGo Zero
**Go formalism:** states are 19×19 boards with history; actions are legal placements or pass; terminal reward \(z\in\{-1,0,1\}\). Go’s combinatorics make raw search infeasible.

**Networks:** A 13-layer convolutional policy network \(p_\sigma(a|s)\) outputs move probabilities, while a structurally similar value network \(v_\theta(s)\) predicts win probability (Fig. 8.4).

**Training:**
- Supervised policy learning from expert games via \(\nabla_\sigma \log p_\sigma(a|s)\) (Eq. 8.39).
- Self-play policy-gradient refinement: \(\Delta \rho \propto \nabla \log p_\rho(a_t|s_t) z_t\) (Eq. 8.40).
- Value network regression on self-play outcomes via \(\nabla_\theta (z-v_\theta(s))^2\) (Eq. 8.41).

**MCTS evolution:** The chapter progressively constructs full MCTS: starting from policy sampling (Eq. 8.42–8.48), adding multi-depth Monte Carlo averages, storing \(N(s,a),W(s,a),Q(s,a)\) per edge (Eq. 8.49–8.52), introducing PUCT selection \(a = \arg\max[Q+ c\,p(a|s)\sqrt{\sum_b N(s,b)}/(1+N(s,a))]\) (Eq. 8.54), expanding nodes dynamically, and backing up leaf values (Eq. 8.55–8.62). Policy priors cut branching breadth; value bootstrapping cuts search depth, achieving tractable Go planning. Full MCTS marries System 2 planning with System 1 networks.

**AlphaGo → AlphaGo Zero:** Original AlphaGo trained networks separately (London) and used them fixed during matches (Seoul). Observing that MCTS outplays raw nets inspires AlphaGo Zero: start from random nets, let MCTS self-play generate data, train policy/value to imitate MCTS moves and outcomes, iterate (Eq. 8.70–8.72). The dual-process view (MCTS as conscious System 2, nets as intuitive System 1) suggests intelligence arises from planning distilled via learning, with planning seen as more fundamental than reinforcement-learning per se.

## 8.9 Deep Q-Learning for Atari
Unlike sparse-reward Go with perfect models, Atari agents operate in dense-reward, model-free regimes using raw pixels as states. DQN approximates \(Q^*(s,a)\) with a CNN, leveraging:
- Experience replay buffers to decorrelate samples.
- Target networks \(Q_{\theta^-}\) updated every C steps (Eq. 8.75–8.77).
- \(\epsilon\)-greedy exploration, frame stacking, reward clipping, gradient clipping.
Despite lacking planning, DQN shares bootstrap principles with MCTS: both propagate leaf/value estimates backward. The chapter notes parallels and contrasts (planning vs learning, value sources, exploration strategies).

## 8.10 Policy Gradients (PPO-style Atari)
Policy-gradient methods directly optimize \(J(\theta) = \mathbb{E}_{\pi_\theta} \sum_t r_t\) via \(\nabla_\theta J = \mathbb{E}[\nabla \log \pi_\theta(a_t|s_t) Q_{\pi}(s_t,a_t)]\) (Eq. 8.80). REINFORCE (Eq. 8.81–8.82) uses trajectory returns; actor-critic variants subtract baselines \(A=R_t - V(s_t)\) (Eq. 8.83) and apply critic updates. Practical Atari implementations use shared CNN encoders, GAE advantages (Eq. 8.84), and on-policy data batches. Compared to Q-learning, policy gradients handle stochastic/continuous actions and often exhibit better stability though lower sample efficiency.

---
Chapter 8 thus reframes flagship results—AlphaGo’s neural MCTS, AlphaGo Zero’s self-play bootstrap, Atari DQN/PPO—as points on the same theoretical landscape, grounded in Bellman operators, policy gradients, and the interplay between model-based planning and function approximation. The System 1/System 2 reflection closes by arguing that planning provides the “conscious” scaffolding which neural networks memorize for fast reactive play.
