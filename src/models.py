import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Standard Expected Q-Network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class IMMQNetwork(nn.Module):
    """Inductive Moment Matching Q-Network (outputs N particles)"""
    def __init__(self, state_dim, action_dim, num_particles=32, hidden_dim=256):
        super().__init__()
        self.num_particles = num_particles
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_particles)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x) # Shape: (batch_size, num_particles)

def compute_mmd(samples1, samples2, sigmas=[0.1, 1.0, 10.0]):
    """
    Computes Maximum Mean Discrepancy using a mixture of RBF kernels.
    samples1: (batch_size, N)
    samples2: (batch_size, M)
    """
    batch_size, n = samples1.shape
    _, m = samples2.shape
    
    s1 = samples1.unsqueeze(2) # (B, n, 1)
    s2 = samples2.unsqueeze(2) # (B, m, 1)
    
    diff_xx = s1 - s1.transpose(1, 2) # (B, n, n)
    diff_yy = s2 - s2.transpose(1, 2) # (B, m, m)
    diff_xy = s1 - s2.transpose(1, 2) # (B, n, m)
    
    dist_xx = diff_xx ** 2
    dist_yy = diff_yy ** 2
    dist_xy = diff_xy ** 2
    
    mmd = 0.0
    for sigma in sigmas:
        k_xx = torch.exp(-dist_xx / (2 * sigma ** 2))
        k_yy = torch.exp(-dist_yy / (2 * sigma ** 2))
        k_xy = torch.exp(-dist_xy / (2 * sigma ** 2))
        
        # Unbiased MMD estimator handles diagonal (self-similarity) differently 
        # but for simplicity and stability, we use the standard empirical MMD:
        mmd_sigma = k_xx.mean(dim=(1, 2)) + k_yy.mean(dim=(1, 2)) - 2 * k_xy.mean(dim=(1, 2))
        mmd += mmd_sigma
        
    return mmd.mean()
