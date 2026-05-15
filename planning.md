## Motivation & Novelty Assessment

### Why This Research Matters
Reinforcement learning often struggles with high variance and instability in complex continuous control tasks like HalfCheetah. Traditional expected Q-learning minimizes the mean squared Bellman error but loses information about the underlying distribution of returns, which is crucial for robust decision-making, exploration, and handling uncertainty.

### Gap in Existing Work
Distributional RL methods (e.g., C51, QR-DQN) require defining restrictive support statistics, such as discrete categories or predefined quantiles, which can introduce approximation errors and complexity in continuous environments. While Moment Matching DQN (MMD-DQN) via MMD has been explored for unstructured distributional RL, combining it with the newer single-stage "Inductive Moment Matching" (IMM) approach (Zhou et al., 2025) remains unexplored. IMM has shown stability in generative modeling and offers an inductive way to perform matching across time steps without heavy distillation processes. 

### Our Novel Contribution
We propose Inductive Moment Matching Q-learning (IMM-Q), which applies the recent IMM framework to the distributional Bellman operator. We match the moments of the predicted action-value distribution to the Bellman target distribution using MMD via inductive bootstrapping.

### Experiment Justification
- **Experiment 1 (Baseline Verification):** Training standard Q-learning and distributional baselines (e.g., standard MMD-DQN or QR-DQN) on HalfCheetah trajectories to establish a performance floor and ensure the environment works.
- **Experiment 2 (IMM-Q Evaluation):** Training our proposed IMM-Q method on varying amounts of trajectories (100, 500, 1000) and comparing sample efficiency, distribution matching stability (MMD loss), and return performance against the baselines.
- **Experiment 3 (Regularization Path):** Evaluating IMM-Q performance against different regularization strengths (alpha) to understand its robustness and to provide the required regularizer path visualizations.

---

## Research Question
Can Inductive Moment Matching (IMM) be effectively integrated into the distributional Bellman update of Q-learning to improve sample efficiency, learning stability, and final performance in continuous control environments?

## Background and Motivation
Q-learning estimates the action-value function but can suffer from high variance and slow convergence. Distributional RL improves upon this by learning the full return distribution. Existing methods like QR-DQN or Categorical DQN impose structural constraints on the learned distribution. IMM provides a more flexible way to match unrestricted moments between the predicted distribution and the target distribution.

## Hypothesis Decomposition
1. **Convergence Rate:** IMM-Q will show faster convergence (higher rewards in fewer gradient steps) compared to baseline Q-learning.
2. **Sample Efficiency:** IMM-Q will outperform standard methods when trained on small dataset sizes (e.g., 100 episodes).
3. **Stability:** The moment matching loss will exhibit lower variance during training than standard temporal difference error.

## Proposed Methodology

### Approach
A controlled comparison study utilizing offline continuous control data (HalfCheetah-v2). Since IMM requires matching continuous distributions, we will use a continuous Q-learning framework, parameterized via neural networks outputting samples (or parameters of a mixture), and update them using an MMD-based loss with inductive bootstrapping.

### Experimental Steps
1. **Initialize and collect/load trajectories:** We will use offline dataset trajectories (from `datasets/` via D4RL) to represent varying dataset sizes.
2. **Implement baselines:** A standard Expected TD Q-learning (or DQN equivalent for continuous space) and an implicit/sample-based DistRL baseline (MMD-DQN or QR-DQN).
3. **Implement IMM-Q:** Develop the Q-network to output a set of samples (or moments) and update them by minimizing the MMD to the Bellman target (reward + gamma * next_state_samples), utilizing an inductive moment matching regularizer/bootstrap.
4. **Train and Evaluate:** Train models on subsets of data (100, 500, 1000 episodes). Track training loss and validation return.

### Baselines
- Temporal Difference Flow / standard Q-learning
- MMD-DQN (Nguyen et al., 2020)
- QR-DQN

### Evaluation Metrics
- **Mean Return (Evaluation):** Primary performance metric on held-out data/environment.
- **MMD Loss:** To track distribution matching stability.
- **Sample Efficiency:** Return achieved per dataset size.

### Statistical Analysis Plan
- Compute mean and standard deviation of evaluation returns across multiple seeds (e.g., 3-5).
- Paired t-tests between IMM-Q and baselines.

## Expected Outcomes
- IMM-Q achieves higher returns on 100-episode datasets than standard Q-learning.
- Training loss curves for IMM-Q are smoother due to the inductive moment matching stabilization.

## Timeline and Milestones
- **Setup (10 mins):** Environment, data loading, dependency installation.
- **Implementation (60 mins):** Baseline and IMM-Q model implementation, training loop.
- **Experimentation (60 mins):** Running training runs across dataset sizes and regularizations.
- **Analysis (30 mins):** Statistical testing, generating plots.
- **Documentation (20 mins):** Finalizing REPORT.md.

## Potential Challenges
- **Computational Cost:** MMD is $O(N^2)$ in the number of samples per state-action pair. We will mitigate this by keeping the sample set small (e.g., $N=32$ or $64$).
- **Environment compatibility:** D4RL HalfCheetah-v2 offline dataset handling. Will need to parse the dataset correctly into transitions.

## Success Criteria
- Successful execution of IMM-Q and baselines without errors.
- Generation of the required plots: (1) Training/val curves, (2) Regularization path, (3) Performance vs regularization strength.
- A comprehensive REPORT.md documenting the findings and statistical comparisons.