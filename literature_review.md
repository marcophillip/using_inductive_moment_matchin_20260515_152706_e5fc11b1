# Literature Review: Using Inductive Moment Matching for Q-learning

## Research Area Overview
This research explores the intersection of **Inductive Moment Matching (IMM)** and **Distributional Reinforcement Learning**. The core idea is to leverage the stability and single-stage training properties of IMM to improve the learning of return distributions in Q-learning. Traditional distributional RL methods (like C51 or QR-DQN) often rely on predefined statistics (categories or quantiles), which can be restrictive. Recent work has proposed using Maximum Mean Discrepancy (MMD) to match unrestricted statistics (moments/samples). IMM offers a potentially more efficient "inductive" way to perform this matching across multiple time steps.

## Key Papers

### Paper 1: Inductive Moment Matching
- **Authors**: Linqi Zhou, Stefano Ermon, Jiaming Song
- **Year**: 2025
- **Source**: arXiv (cs.LG) / PMLR 267
- **Key Contribution**: Proposes IMM, a single-stage training procedure for generative models that learns a one-step sampler via inductive bootstrapping.
- **Methodology**: Uses marginal-preserving interpolants and MMD-based objectives to ensure distribution-level convergence without needing distillation from a teacher model.
- **Relevance to Our Research**: Provides the core "Inductive" and "Moment Matching" framework that we aim to apply to the Bellman operator in RL.

### Paper 2: Distributional Reinforcement Learning via Moment Matching
- **Authors**: Thanh Tang Nguyen, Sunil Gupta, Svetha Venkatesh
- **Year**: 2020 (v3)
- **Source**: AAAI 2021
- **Key Contribution**: Formulates a distributional RL method that learns unrestricted statistics (deterministic samples) by matching moments using MMD.
- **Methodology**: Minimizes the MMD between the current return distribution and the Bellman target distribution. Proves the contraction property of the distributional Bellman operator under MMD.
- **Results**: Outperforms C51 and QR-DQN on Atari games.
- **Relevance to Our Research**: Demonstrates that moment matching via MMD is a viable and powerful approach for distributional RL.

### Paper 3: A Distributional Perspective on Reinforcement Learning
- **Authors**: Marc G. Bellemare, Will Dabney, Rémi Munos
- **Year**: 2017
- **Source**: ICML 2017
- **Key Contribution**: Introduces the concept of learning the full distribution of returns (Categorical DQN / C51) rather than just the expectation.
- **Relevance**: Foundational paper for all distributional RL research.

### Paper 4: Implicit Quantile Networks for Distributional Reinforcement Learning
- **Authors**: Will Dabney, Georg Ostrovski, David Silver, Rémi Munos
- **Year**: 2018
- **Source**: ICML 2018
- **Key Contribution**: Proposes IQN, which learns a quantile function using a re-parameterization trick, allowing for an implicit representation of the return distribution.
- **Relevance**: A state-of-the-art baseline for distributional RL that uses implicit samples.

## Common Methodologies
- **Distributional Bellman Operator**: Updating the distribution $Z(s, a)$ instead of the scalar $Q(s, a)$.
- **Maximum Mean Discrepancy (MMD)**: A kernel-based method to measure the distance between distributions, used in both Nguyen et al. (2020) and Zhou et al. (2025).
- **Inductive Bootstrapping**: Using previous estimates (or intermediate time-steps) to guide the learning of the current distribution, a key feature of IMM.

## Standard Baselines
- **DQN**: Standard expected RL.
- **C51**: Categorical Distributional RL.
- **QR-DQN**: Quantile Regression DQN.
- **IQN**: Implicit Quantile Networks.
- **MMD-DQN**: (The method from Nguyen et al. 2020).

## Evaluation Metrics
- **Mean Episode Reward**: Standard RL performance.
- **Sample Efficiency**: Learning speed (reward vs. steps).
- **Calibration/Uncertainty**: How well the learned distribution captures true return variance.
- **MMD**: As a measure of convergence between the learned and true distributions.

## Datasets in the Literature
- **Atari 2600**: Discrete action space benchmarks.
- **Mujoco (HalfCheetah, Hopper, etc.)**: Continuous control benchmarks (relevant to the user's `half-cheetah-v2` mention).
- **D4RL**: Offline RL datasets for Mujoco environments.

## Gaps and Opportunities
- **Stability of Moment Matching**: While MMD-DQN is powerful, combining it with IMM's "inductive" approach might provide better stability and faster convergence in complex continuous environments like HalfCheetah.
- **Exploration**: Using the full distribution from IMM-RL to derive uncertainty-aware exploration policies (e.g., via Thompson Sampling or UCB on moments).

## Recommendations for Our Experiment
1. **Primary Dataset**: `half-cheetah-v2` (D4RL medium/expert datasets for offline, or online Gym environment).
2. **Baseline Methods**: QR-DQN, IQN, and MMD-DQN.
3. **Core Task**: Implement an "IMM-Q" agent that uses the Inductive Moment Matching objective to update its value distribution.
4. **Metrics**: Reward curves, MMD to Bellman target, and uncertainty estimation accuracy.
