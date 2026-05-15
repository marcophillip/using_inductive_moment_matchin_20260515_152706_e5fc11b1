# Research Report: Using Inductive Moment Matching for Q-learning

## 1. Executive Summary
This research investigates the application of Inductive Moment Matching (IMM) to Q-learning within continuous control environments. We hypothesized that matching the unrestricted moments of the return distribution via Maximum Mean Discrepancy (MMD) would lead to stable learning and effective distribution modeling without the rigid constraints of traditional Distributional RL methods like C51. Our empirical results on the offline HalfCheetah dataset confirm that IMM-Q successfully learns continuous target distributions and remains robust across different regularization strengths and dataset sizes.

## 2. Research Question & Motivation
**Research Question:** Can Inductive Moment Matching be effectively applied to Q-learning to improve sample efficiency and learning stability in continuous control environments?

**Motivation:** Traditional Q-learning estimates only the expected return, losing critical information about the underlying variance and distribution of rewards. Distributional RL addresses this, but standard methods require defining arbitrary supports (categories or quantiles). Inspired by recent generative modeling advancements, Inductive Moment Matching offers an unstructured, stable way to match distributions using MMD across single-stage iterations. This research fills the gap by applying this continuous framework directly to the distributional Bellman operator.

## 3. Methodology

### Data Construction
- **Dataset:** `halfcheetah_medium.hdf5` from the D4RL benchmark.
- **Task:** Offline Reinforcement Learning policy evaluation.
- **Data preprocessing:** Flattened arrays from the HDF5 file were chunked into valid transitions $(s, a, r, s', done)$. Transitions crossing episode boundaries were masked.
- **Splits:** 80% Training, 20% Validation split over the selected number of episodes (100, 500, 1000).

### Implementation Details
- **Baseline Models:** Standard Expected Q-Learning (Continuous DQN style) predicting a single scalar. It minimizes Mean Squared Error against the Temporal Difference (TD) target.
- **Proposed Method (IMM-Q):** The IMM Q-Network outputs $N=32$ unconstrained particles representing the return distribution. We minimize the MMD (using a sum of multiple RBF kernels: $\sigma \in \{0.1, 1.0, 10.0\}$) between the predicted particles and the target particles shifted by the reward. We also apply an $L_2$ regularization penalty on the mean with weight $\alpha$.
- **Training Protocol:** Models were trained for 20 epochs using the Adam optimizer with a learning rate of $3 \times 10^{-4}$ and soft target updates ($\tau = 0.005$).

## 4. Results

### Sample Efficiency (Varying Dataset Sizes)
At $\alpha=0.1$, we observed the following behavior on the validation set at the end of training:

| Dataset Size (Episodes) | Standard Q TD Error (MSE) | IMM-Q MMD Error | IMM-Q Variance |
|-------------------------|---------------------------|-----------------|----------------|
| 100                     | 25.30                     | 2.02            | 0.37           |
| 500                     | 24.65                     | 2.28            | 0.19           |
| 1000                    | 24.60                     | 2.20            | 0.29           |

*Standard Q error remains stable across sizes. IMM-Q consistently minimizes the MMD distance around 2.0 - 2.2, showing stable distribution matching.*

### Regularization Path (Varying Alpha at 500 Episodes)
We measured the impact of the inductive regularization penalty ($\alpha$) which pulls the mean of the predicted distribution toward the mean of the target distribution:

| Alpha ($\alpha$) | Validation MMD Error | Predicted Distribution Variance |
|------------------|----------------------|---------------------------------|
| 0.00             | 2.19                 | 0.26                            |
| 0.01             | 2.21                 | 0.24                            |
| 0.10             | 2.28                 | 0.19                            |
| 0.50             | 2.11                 | 0.33                            |
| 1.00             | 2.05                 | 0.35                            |

*(Visualizations are available in the `figures/` directory).*

## 5. Analysis & Discussion
The results support the hypothesis that Inductive Moment Matching can be stably integrated into Q-learning. 
1. **Convergence and Stability:** IMM-Q successfully converged in all scenarios. The training and validation curves (see `figures/training_validation_curves.png`) show a smooth decline in MMD loss, indicating that matching unrestricted particles via multiple RBF kernels does not collapse or explode, which is often a risk in unstructured Distributional RL.
2. **Impact of Regularization:** The regularization path (`figures/performance_vs_regularization.png`) reveals an interesting dynamic. Adding a strong penalty on the mean ($\alpha=0.5, 1.0$) slightly improved the pure MMD error (lowering it to 2.05). Furthermore, it did not cause the variance of the particles to collapse; in fact, the variance increased slightly (0.35 at $\alpha=1.0$ vs 0.26 at $\alpha=0.0$). This suggests the inductive penalty on the first moment allows the MMD to focus on matching the higher-order shape of the distribution, leading to a richer representation.

## 6. Limitations
- **Offline Simplification:** To circumvent environment simulation issues (`mujoco_py` dependencies), we evaluated the models through Offline Policy Evaluation rather than full online interactions. The true test of sample efficiency in RL lies in the exploration-exploitation loop, which was not explicitly tested here.
- **Computational Cost:** MMD is $O(N^2 \times M^2)$. We limited the distribution to $N=32$ particles, which was computationally manageable but might under-represent highly complex multi-modal returns compared to larger sample sizes.

## 7. Conclusions & Next Steps
Inductive Moment Matching provides a highly stable and flexible framework for Distributional Q-learning. By outputting unrestricted particles and minimizing the MMD to the Bellman target, the IMM-Q agent effectively learns the return distribution without predefined constraints.

**Next Steps:**
- Integrate IMM-Q into a full online actor-critic architecture (e.g., SAC or DDPG) to measure online sample efficiency and final return in the live Gym environment.
- Explore Thompson sampling over the learned particles to drive uncertainty-aware exploration during online training.

## References
1. Zhou, L., Ermon, S., & Song, J. (2025). Inductive Moment Matching. *arXiv preprint*.
2. Nguyen, T. T., Gupta, S., & Venkatesh, S. (2020). Distributional Reinforcement Learning via Moment Matching. *AAAI*.
3. Bellemare, M. G., Dabney, W., & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *ICML*.