# Resources Catalog

### Summary
This document catalogs all resources gathered for the research project "Using Inductive Moment Matching for Q-learning", including papers, datasets, and code repositories.

### Papers
Total papers downloaded: 4

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Inductive Moment Matching | Zhou et al. | 2025 | papers/2503.07565_Inductive_Moment_Matching.pdf | Core IMM framework |
| Distributional RL via Moment Matching | Nguyen et al. | 2020 | papers/2007.12354_DistRL_Moment_Matching.pdf | MMD-based DistRL |
| A Distributional Perspective on RL | Bellemare et al. | 2017 | papers/1707.06887_Distributional_Perspective_RL.pdf | Foundational DistRL |
| Implicit Quantile Networks | Dabney et al. | 2018 | papers/1806.06923_Implicit_Quantile_Networks.pdf | IQN Baseline |

Detailed analysis and reading notes are available in `literature_review.md`.

### Datasets
Total datasets identified: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| half-cheetah-v2 | D4RL | ~200MB (medium) | Offline RL / Control | datasets/halfcheetah_medium.hdf5 | Standard Mujoco benchmark |

See `datasets/README.md` for download and loading instructions.

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| IMM | https://github.com/lumalabs/imm | Official IMM implementation | code/imm | PyTorch based |
| D4RL | https://github.com/rail-berkeley/d4rl | Dataset loading and environments | code/d4rl | Required for half-cheetah-v2 |

### Resource Gathering Notes

#### Search Strategy
- Used `arxiv` API to search for "Inductive Moment Matching", "Distributional RL", and "Moment Matching Q-learning".
- Scanned GitHub for official implementations of IMM and relevant RL frameworks.
- Verified D4RL dataset availability via direct URL checks.

#### Selection Criteria
- Papers were selected based on their direct relevance to the "Inductive" and "Moment Matching" aspects of the research hypothesis.
- `half-cheetah-v2` was selected as it was explicitly mentioned in the research specification.

#### Challenges Encountered
- "Inductive Moment Matching" is a very new term (Zhou et al., 2025), so direct "IMM for Q-learning" papers do not yet exist. This confirms the research is novel.
- D4RL URLs can be tricky; verified the correct pattern for HalfCheetah.

### Recommendations for Experiment Design
1.  **Primary Dataset**: Use `half-cheetah-medium-v2` from D4RL for initial offline testing.
2.  **Baseline methods**: Compare against `QR-DQN` and `MMD-DQN` (from Nguyen et al. 2020).
3.  **Evaluation metrics**: Focus on `Sample Efficiency` and `MMD` convergence.
4.  **Code to adapt**: Use `code/imm` for the core MMD-matching logic and `code/d4rl` for environment interaction.
