import json
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    with open("../results/metrics.json", "r") as f:
        return json.load(f)

def plot_learning_curves(results):
    plt.figure(figsize=(12, 5))
    
    # Take 500 episodes result for alpha 0.1
    res = next(r for r in results if r['episodes'] == 500 and r['alpha'] == 0.1)
    
    plt.subplot(1, 2, 1)
    epochs = np.arange(1, len(res['q_train_losses']) + 1)
    plt.plot(epochs, res['q_train_losses'], label='Q Train', color='blue')
    plt.plot(epochs, res['q_val_losses'], label='Q Val', color='blue', linestyle='--')
    plt.title('Standard Q-Learning Loss (TD Error)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, res['imm_train_losses'], label='IMM Train', color='orange')
    plt.plot(epochs, res['imm_val_losses'], label='IMM Val', color='orange', linestyle='--')
    plt.title('IMM Q-Learning Loss (MMD)')
    plt.xlabel('Epochs')
    plt.ylabel('MMD')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../figures/training_validation_curves.png")
    plt.close()

def plot_performance_vs_alpha(results):
    alphas = [0.0, 0.01, 0.1, 0.5, 1.0]
    
    mmd_errors = []
    variances = []
    
    for a in alphas:
        # Find result for 500 episodes
        res = next(r for r in results if r['episodes'] == 500 and r['alpha'] == a)
        mmd_errors.append(res['metrics']['imm_q_mmd_error'])
        variances.append(res['metrics']['imm_q_variance'])
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(alphas, mmd_errors, marker='o', color='purple')
    plt.title('Validation MMD Error vs Regularization (Alpha)')
    plt.xlabel('Alpha (L2 Regularization Strength)')
    plt.ylabel('MMD Error on Held-Out Set')
    plt.xscale('symlog', linthresh=0.01)
    
    plt.subplot(1, 2, 2)
    plt.plot(alphas, variances, marker='s', color='green')
    plt.title('Distribution Variance vs Regularization (Alpha)')
    plt.xlabel('Alpha')
    plt.ylabel('Average Variance of Particles')
    plt.xscale('symlog', linthresh=0.01)
    
    plt.tight_layout()
    plt.savefig("../figures/performance_vs_regularization.png")
    plt.close()

def plot_regularization_path(results):
    # As a proxy for regularization path, we plot the effect on final train loss and val loss
    alphas = [0.0, 0.01, 0.1, 0.5, 1.0]
    final_train = []
    final_val = []
    
    for a in alphas:
        res = next(r for r in results if r['episodes'] == 500 and r['alpha'] == a)
        final_train.append(res['imm_train_losses'][-1])
        final_val.append(res['imm_val_losses'][-1])
        
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, final_train, marker='^', label='Final Train Loss', color='brown')
    plt.plot(alphas, final_val, marker='v', label='Final Val Loss', color='teal')
    plt.title('Regularization Path (Alpha vs Final Losses)')
    plt.xlabel('Alpha')
    plt.ylabel('MMD + Alpha Penalty Loss')
    plt.xscale('symlog', linthresh=0.01)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../figures/regularization_path.png")
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("../figures", exist_ok=True)
    results = load_results()
    plot_learning_curves(results)
    plot_performance_vs_alpha(results)
    plot_regularization_path(results)
    print("Plots saved in ../figures/")
