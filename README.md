# Project Overview: Using Inductive Moment Matching for Q-learning

## Description
This project investigates the application of Inductive Moment Matching (IMM) to Q-learning (IMM-Q) to improve stability and sample efficiency in continuous control tasks. By learning the distribution of returns and matching their moments using Maximum Mean Discrepancy (MMD) with inductive bootstrapping, IMM-Q provides a flexible, unstructured approach to Distributional Reinforcement Learning.

## Key Findings
- **Distributional Stability:** IMM-Q smoothly minimizes the Maximum Mean Discrepancy (MMD) to the target Bellman distribution across all tested dataset sizes (100, 500, 1000 episodes).
- **Regularization Robustness:** The L2 regularization parameter (alpha) successfully regularizes the mean values without destroying the variance of the learned distribution. At $\alpha=1.0$, the particle variance slightly increased compared to $\alpha=0.0$, demonstrating that inductive moment matching maintains diversity in the predicted distribution.
- **Offline Evaluation Performance:** Both standard Q-learning and IMM-Q achieve stable convergence on the D4RL `halfcheetah_medium` offline dataset. IMM-Q effectively models the return distribution without requiring predefined supports or quantiles.

## How to Reproduce
1. Ensure you have python 3.10+ and `uv` installed.
2. Activate the virtual environment: `source .venv/bin/activate`
3. The HalfCheetah dataset should be downloaded to `datasets/halfcheetah_medium.hdf5`.
4. Run the training script: `cd src && python train.py`
5. Generate the plots: `cd src && python plot.py`

## File Structure Overview
- `datasets/`: Contains the D4RL `halfcheetah_medium.hdf5` dataset.
- `src/`: Python source code containing data loaders (`data.py`), neural networks (`models.py`), training loop (`train.py`), and plot generation (`plot.py`).
- `results/`: Contains the raw JSON metrics output from the experiments.
- `figures/`: Contains the generated PNG plots from the analysis.
- `REPORT.md`: Comprehensive research report detailing motivation, methodology, and empirical findings.

For a full scientific discussion, refer to [REPORT.md](./REPORT.md).