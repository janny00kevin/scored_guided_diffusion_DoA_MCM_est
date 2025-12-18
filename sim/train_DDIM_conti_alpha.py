import torch
import torch.optim as optim
import os
import sys
from models.epsnet_mlp import EpsNetMLP

try:
    from training_all_snapshots_same_time_PINN_MCM_proj_steering_vec.diffusion.continuous_beta import alpha_bar_of_t
except ImportError as e:
    print(f"Error importing continuous_beta. Make sure the module path is correct.")
    raise e

# =============================
# Configuration
# =============================
# 
TRAIN_DATA_PATH = "dataset/training_dataset.pt"

NUM_EPOCHS = 50
BATCH_SIZE = 4096
LR = 1e-3
T = 1000  # Diffusion steps
T_continuous = 1000.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Helpers: Diffusion Schedule
# =============================
# DDIM/DDPM schedule
beta_min = 1e-4
beta_max = 0.02
betas = torch.linspace(beta_min, beta_max, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_1m_alpha_bars = torch.sqrt(1.0 - alpha_bars)
MODEL_SAVE_PATH = f"weights/DDIM_ep{NUM_EPOCHS}_lr{LR:.0e}_t{T}_bmax{beta_max:.0e}_nmlz_.pth"

def complex_to_real(x):
    # x: (..., N) -> (..., 2N)
    return torch.cat([x.real, x.imag], dim=-1)

# =============================
# Training Loop
# =============================
def train():
    if not os.path.exists("weights"):
        os.makedirs("weights")
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: {TRAIN_DATA_PATH} not found. Please run generate_train_dataset.py first.")
        return

    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    Xs = torch.load(TRAIN_DATA_PATH).to(device) # (S, N, L)
    
    # Flatten to (Total_Snapshots, N)
    S, N, L = Xs.shape
    X_flat = Xs.permute(0, 2, 1).reshape(-1, N)
    X_real = complex_to_real(X_flat) # (S*L, 2N)

    # Compute mean and std for normalization
    data_mean = torch.mean(X_real)#torch.tensor(0.0, device=device)
    data_std = torch.std(X_real)#torch.tensor(1.0, device=device)
    data_std[data_std < 1e-8] = 1.0 # Prevent division by zero
    X_real = (X_real - data_mean) / data_std
    
    dataset_size = X_real.shape[0]
    print(f"Total training snapshots: {dataset_size}")

    # Initialize EpsNet (MLP as per your original main.py)
    dim = 2 * N
    eps_net = EpsNetMLP(dim=dim, hidden=1024, time_emb_dim=128).to(device)
    optimizer = optim.Adam(eps_net.parameters(), lr=LR)
    
    print("Starting training (EpsNet for DDIM)...")
    eps_net.train()
    
    for epoch in range(NUM_EPOCHS):
        indices = torch.randperm(dataset_size, device=device)
        total_loss = 0.0
        num_batches = 0
        
        for start_idx in range(0, dataset_size, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, dataset_size)
            batch_idx = indices[start_idx:end_idx]
            x0_batch = X_real[batch_idx] # (B, 2N)
            current_bs = x0_batch.shape[0]
            
            # 1. Sample t
            # t = torch.randint(0, T, (current_bs,), device=device)
            t_cont = torch.rand(current_bs, device=device) * T_continuous

            a_bar = alpha_bar_of_t(t_cont, beta_min=beta_min, beta_max=beta_max, T=T_continuous)
            
            sqrt_a = torch.sqrt(a_bar).view(-1, 1)
            sqrt_1ma = torch.sqrt(1.0 - a_bar).view(-1, 1)
            
            # 2. Add Noise (Forward)
            noise = torch.randn_like(x0_batch)
            # a_bar = sqrt_alpha_bars[t].unsqueeze(1)
            # one_minus_a_bar = sqrt_1m_alpha_bars[t].unsqueeze(1)
            
            # x_t = a_bar * x0_batch + one_minus_a_bar * noise
            x_t = sqrt_a * x0_batch + sqrt_1ma * noise
            
            # 3. Predict Noise (Eps)
            # Input t needs to be continuous-like or normalized for the network
            pred_eps = eps_net(x_t, t_cont)
            
            loss = torch.mean((pred_eps - noise) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")
            
    checkpoint = {
        'model_state_dict': eps_net.state_dict(),
        'data_mean': data_mean,
        'data_std': data_std,
        'config': {
            'T': T,
            'beta_min': beta_min,
            'beta_max': beta_max
        }
    }
    
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"Model and statistics saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()