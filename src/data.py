import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class D4RLDataset(Dataset):
    def __init__(self, hdf5_path, num_episodes=None, split="train", seed=42):
        self.hdf5_path = hdf5_path
        self.num_episodes = num_episodes
        self.split = split
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # D4RL stores flat arrays. We need to split them into episodes using terminals/timeouts.
            obs = f['observations'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            terminals = f['terminals'][:]
            
            # Handle timeout if it exists
            if 'timeouts' in f:
                timeouts = f['timeouts'][:]
            else:
                timeouts = np.zeros_like(terminals)
                
        # Find episode boundaries
        episode_ends = np.where(terminals | timeouts)[0]
        
        # Build transitions
        self.states = obs[:-1]
        self.actions = actions[:-1]
        self.rewards = rewards[:-1]
        self.next_states = obs[1:]
        self.dones = terminals[:-1]
        
        # Mask out transitions that cross episode boundaries
        # If step i is the end of an episode, then (state_i, act_i) -> next_state is NOT valid.
        valid_transitions = np.ones(len(self.states), dtype=bool)
        for end_idx in episode_ends:
            if end_idx < len(valid_transitions):
                valid_transitions[end_idx] = False
                
        self.states = self.states[valid_transitions]
        self.actions = self.actions[valid_transitions]
        self.rewards = self.rewards[valid_transitions]
        self.next_states = self.next_states[valid_transitions]
        self.dones = self.dones[valid_transitions]
        
        # Limit dataset by num_episodes if specified
        # To do this roughly, we can compute average episode length
        avg_ep_len = len(obs) / (len(episode_ends) + 1)
        if self.num_episodes is not None:
            max_transitions = int(self.num_episodes * avg_ep_len)
            self.states = self.states[:max_transitions]
            self.actions = self.actions[:max_transitions]
            self.rewards = self.rewards[:max_transitions]
            self.next_states = self.next_states[:max_transitions]
            self.dones = self.dones[:max_transitions]
            
        # Train / test split (80/20)
        np.random.seed(seed)
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        # Convert to torch tensors
        self.states = torch.FloatTensor(self.states[self.indices])
        self.actions = torch.FloatTensor(self.actions[self.indices])
        self.rewards = torch.FloatTensor(self.rewards[self.indices]).unsqueeze(-1)
        self.next_states = torch.FloatTensor(self.next_states[self.indices])
        self.dones = torch.FloatTensor(self.dones[self.indices]).unsqueeze(-1)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'done': self.dones[idx]
        }

def get_dataloaders(hdf5_path, batch_size=256, num_episodes=None):
    train_dataset = D4RLDataset(hdf5_path, num_episodes=num_episodes, split="train")
    val_dataset = D4RLDataset(hdf5_path, num_episodes=num_episodes, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
