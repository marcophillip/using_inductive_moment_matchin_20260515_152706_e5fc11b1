import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import argparse
from tqdm import tqdm

from data import get_dataloaders
from models import QNetwork, IMMQNetwork, compute_mmd

def get_random_actions(batch_size, action_dim, num_samples=10, device='cuda'):
    # HalfCheetah actions are typically in [-1, 1]
    return torch.rand(batch_size, num_samples, action_dim, device=device) * 2.0 - 1.0

def train_standard_q(train_loader, val_loader, state_dim, action_dim, num_epochs=50, device='cuda', gamma=0.99):
    model = QNetwork(state_dim, action_dim).to(device)
    target_model = QNetwork(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            reward = batch['reward'].to(device)
            next_state = batch['next_state'].to(device)
            done = batch['done'].to(device)
            
            # Compute target
            with torch.no_grad():
                # Randomly sample actions to estimate max Q
                random_actions = get_random_actions(state.shape[0], action_dim, num_samples=10, device=device)
                # repeat next_state for each random action
                repeated_next_state = next_state.unsqueeze(1).repeat(1, 10, 1)
                
                # evaluate target
                next_q = target_model(repeated_next_state, random_actions) # (B, 10, 1)
                max_next_q, _ = torch.max(next_q, dim=1) # (B, 1)
                
                target_q = reward + gamma * (1 - done) * max_next_q
                
            current_q = model(state, action)
            loss = criterion(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Soft update target
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)
            
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                action = batch['action'].to(device)
                reward = batch['reward'].to(device)
                next_state = batch['next_state'].to(device)
                done = batch['done'].to(device)
                
                random_actions = get_random_actions(state.shape[0], action_dim, num_samples=10, device=device)
                repeated_next_state = next_state.unsqueeze(1).repeat(1, 10, 1)
                next_q = target_model(repeated_next_state, random_actions)
                max_next_q, _ = torch.max(next_q, dim=1)
                target_q = reward + gamma * (1 - done) * max_next_q
                
                current_q = model(state, action)
                loss = criterion(current_q, target_q)
                val_loss += loss.item()
                
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"Standard Q | Epoch {epoch+1:02d} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
            
    return model, train_losses, val_losses

def train_imm_q(train_loader, val_loader, state_dim, action_dim, num_epochs=50, device='cuda', gamma=0.99, alpha=0.1):
    model = IMMQNetwork(state_dim, action_dim, num_particles=32).to(device)
    target_model = IMMQNetwork(state_dim, action_dim, num_particles=32).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            reward = batch['reward'].to(device)
            next_state = batch['next_state'].to(device)
            done = batch['done'].to(device)
            
            with torch.no_grad():
                # For distributional RL, max over actions requires comparing distributions.
                # A simple approximation: compare means of the distributions.
                random_actions = get_random_actions(state.shape[0], action_dim, num_samples=10, device=device)
                repeated_next_state = next_state.unsqueeze(1).repeat(1, 10, 1)
                
                # next_dist: (B, 10, num_particles)
                next_dist = target_model(repeated_next_state, random_actions)
                mean_next_q = next_dist.mean(dim=2) # (B, 10)
                
                # find best action index
                best_action_idx = torch.argmax(mean_next_q, dim=1) # (B,)
                
                # gather the distribution for the best action
                # best_action_idx shape (B,) -> we want to pick from (B, 10, num_particles)
                best_next_dist = next_dist[torch.arange(state.shape[0]), best_action_idx] # (B, num_particles)
                
                # target distribution
                target_dist = reward + gamma * (1 - done) * best_next_dist
                
            current_dist = model(state, action)
            
            # MMD loss
            loss_mmd = compute_mmd(current_dist, target_dist)
            
            # IMM often uses an inductive regularizer (alpha) pulling current dist to previous iteration
            # or matching moments. Since we use a target network, the MMD to target is already an inductive step.
            # We can add an L2 penalty on the mean to stabilize if alpha > 0.
            loss = loss_mmd + alpha * torch.mean((current_dist.mean(dim=1, keepdim=True) - target_dist.mean(dim=1, keepdim=True)) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Soft update
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)
            
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                action = batch['action'].to(device)
                reward = batch['reward'].to(device)
                next_state = batch['next_state'].to(device)
                done = batch['done'].to(device)
                
                random_actions = get_random_actions(state.shape[0], action_dim, num_samples=10, device=device)
                repeated_next_state = next_state.unsqueeze(1).repeat(1, 10, 1)
                next_dist = target_model(repeated_next_state, random_actions)
                mean_next_q = next_dist.mean(dim=2)
                best_action_idx = torch.argmax(mean_next_q, dim=1)
                best_next_dist = next_dist[torch.arange(state.shape[0]), best_action_idx]
                target_dist = reward + gamma * (1 - done) * best_next_dist
                
                current_dist = model(state, action)
                val_loss += compute_mmd(current_dist, target_dist).item()
                
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"IMM Q      | Epoch {epoch+1:02d} | Train Loss: {train_losses[-1]:.4f} | Val MMD Loss: {val_losses[-1]:.4f}")
            
    return model, train_losses, val_losses

def evaluate_models(q_model, imm_model, val_loader, state_dim, action_dim, device='cuda'):
    q_model.eval()
    imm_model.eval()
    
    # We will evaluate how well they estimate the Q-value.
    # Since we don't have true Q-values, we can evaluate TD error and MMD error on the validation set.
    # We will also compute variance of predictions.
    
    metrics = {
        'std_q_td_error': 0.0,
        'imm_q_mmd_error': 0.0,
        'imm_q_variance': 0.0
    }
    
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            reward = batch['reward'].to(device)
            next_state = batch['next_state'].to(device)
            done = batch['done'].to(device)
            
            # Standard Q
            random_actions = get_random_actions(state.shape[0], action_dim, num_samples=10, device=device)
            repeated_next_state = next_state.unsqueeze(1).repeat(1, 10, 1)
            next_q = q_model(repeated_next_state, random_actions)
            max_next_q, _ = torch.max(next_q, dim=1)
            target_q = reward + 0.99 * (1 - done) * max_next_q
            current_q = q_model(state, action)
            metrics['std_q_td_error'] += nn.MSELoss()(current_q, target_q).item() * state.shape[0]
            
            # IMM Q
            next_dist = imm_model(repeated_next_state, random_actions)
            mean_next_q = next_dist.mean(dim=2)
            best_action_idx = torch.argmax(mean_next_q, dim=1)
            best_next_dist = next_dist[torch.arange(state.shape[0]), best_action_idx]
            target_dist = reward + 0.99 * (1 - done) * best_next_dist
            current_dist = imm_model(state, action)
            
            metrics['imm_q_mmd_error'] += compute_mmd(current_dist, target_dist).item() * state.shape[0]
            metrics['imm_q_variance'] += current_dist.var(dim=1).mean().item() * state.shape[0]
            
            total += state.shape[0]
            
    for k in metrics:
        metrics[k] /= total
        
    return metrics

def run_experiment(hdf5_path, episodes, alpha, device='cuda'):
    print(f"\n--- Running Experiment: Episodes={episodes}, Alpha={alpha} ---")
    train_loader, val_loader = get_dataloaders(hdf5_path, batch_size=256, num_episodes=episodes)
    
    # HalfCheetah dims
    state_dim = 17
    action_dim = 6
    
    print("Training Standard Q-Network...")
    q_model, q_train, q_val = train_standard_q(train_loader, val_loader, state_dim, action_dim, num_epochs=20, device=device)
    
    print("Training IMM Q-Network...")
    imm_model, imm_train, imm_val = train_imm_q(train_loader, val_loader, state_dim, action_dim, num_epochs=20, device=device, alpha=alpha)
    
    print("Evaluating...")
    metrics = evaluate_models(q_model, imm_model, val_loader, state_dim, action_dim, device=device)
    print(f"Metrics: {metrics}")
    
    return {
        'episodes': episodes,
        'alpha': alpha,
        'q_train_losses': q_train,
        'q_val_losses': q_val,
        'imm_train_losses': imm_train,
        'imm_val_losses': imm_val,
        'metrics': metrics
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/halfcheetah_medium.hdf5")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    
    # Experiment 2: Varying dataset sizes
    for episodes in [100, 500, 1000]:
        res = run_experiment(args.data_path, episodes=episodes, alpha=0.1, device=device)
        results.append(res)
        
    # Experiment 3: Regularization path (using 500 episodes)
    for alpha in [0.0, 0.01, 0.5, 1.0]:
        res = run_experiment(args.data_path, episodes=500, alpha=alpha, device=device)
        results.append(res)
        
    os.makedirs("../results", exist_ok=True)
    with open("../results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Training complete! Results saved to results/metrics.json")
