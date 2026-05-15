# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset: half-cheetah-v2 (D4RL)

### Overview
- **Source**: [D4RL (Berkeley)](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5)
- **Size**: ~200 MB
- **Format**: HDF5
- **Task**: Offline Reinforcement Learning / Continuous Control
- **Environment**: HalfCheetah-v2 (Mujoco)

### Download Instructions

**Using wget (direct download):**
```bash
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5 -O datasets/halfcheetah_medium.hdf5
```

**Using D4RL library (Python):**
```python
import gym
import d4rl # Import registers the environments
env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()
```

### Loading the Dataset

Once downloaded, load with:
```python
import h5py
data = h5py.File('datasets/halfcheetah_medium.hdf5', 'r')
# access data: data['observations'], data['actions'], etc.
```

### Sample Data
The dataset contains transitions:
- `observations`: current state
- `actions`: action taken
- `next_observations`: next state
- `rewards`: reward received
- `terminals`: whether the episode ended

### Notes
- Requires `mujoco` and `gym` for environment simulation.
- `halfcheetah-medium-v2` is typically used for benchmarking sample efficiency and optimality in offline RL.
