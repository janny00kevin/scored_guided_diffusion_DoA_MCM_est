import torch
import torch.optim as optim
import os
from models.score_net import ScoreNet

# =============================
# Configuration
# =============================
TRAIN_DATA_PATH = "dataset/training_dataset.pt"
# MODEL_SAVE_PATH = "weights/DDPM.pth"
NUM_EPOCHS = 1000
BATCH_SIZE = 4096  # Number of columns (snapshots) per batch
LR = 1e-3
T = 50
BETA_MIN = 1e-4
BETA_MAX = 0.02
MODEL_SAVE_PATH = f"weights/DDPM_ep{NUM_EPOCHS}_lr{LR:.0e}_t{T}_bmin{BETA_MIN:.0e}_bmax{BETA_MAX:.2f}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Helpers
# =============================
def complex_to_real(x):
    # x: (..., N) complex -> (..., 2N) real
    return torch.cat([x.real, x.imag], dim=-1)

# Precompute diffusion schedule
# The forward diffusion process adds noise according to this schedule.
betas = torch.linspace(BETA_MIN, BETA_MAX, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_1m_alpha_bars = torch.sqrt(1.0 - alpha_bars)

def q_sample(x0_real, t_idx):
    # Samples the state x_t from the clean data x_0 at time step t_idx
    # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    a_bar = sqrt_alpha_bars[t_idx]
    noise = torch.randn_like(x0_real)
    x_t = a_bar.unsqueeze(1) * x0_real + sqrt_1m_alpha_bars[t_idx].unsqueeze(1) * noise
    return x_t, noise

# =============================
# Training Loop
# =============================
def train():
    if not os.path.exists("weights"):
        os.makedirs("weights")

    # Load Data
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    # Xs shape: (S, N, L) where S is num samples, N is array size, L is num snapshots
    Xs = torch.load(TRAIN_DATA_PATH).to(device)
    S_samples, N, L = Xs.shape
    
    # Flatten to columns for training: (S*L, N)
    print("Preprocessing data...")
    # Permute to (S, L, N) then flatten to (S*L, N)
    # This gives us S*L snapshots for training
    X_flat = Xs.permute(0, 2, 1).reshape(-1, N) 
    X_real = complex_to_real(X_flat) # (Total_Snapshots, 2N)
    
    dataset_size = X_real.shape[0]
    print(f"Total training snapshots (columns): {dataset_size}")

    # Model setup
    dim = 2 * N
    score_net = ScoreNet(dim=dim).to(device)
    optimizer = optim.Adam(score_net.parameters(), lr=LR)
    
    print("Starting training...")
    score_net.train()
    
    # The training objective minimizes the difference between the predicted score
    # and the true score function of the diffusion kernel q(x_t|x_0).
    # 
    
    for epoch in range(NUM_EPOCHS):
        # Shuffle indices
        indices = torch.randperm(dataset_size, device=device)
        total_loss = 0.0
        num_batches = 0
        
        for start_idx in range(0, dataset_size, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, dataset_size)
            batch_idx = indices[start_idx:end_idx]
            
            x0_batch = X_real[batch_idx] # (B, 2N)
            current_bs = x0_batch.shape[0]
            
            # Sample time steps
            t = torch.randint(0, T, (current_bs,), device=device)
            
            # Add noise (Forward Process)
            x_t, noise = q_sample(x0_batch, t)
            
            # Predict Score
            # The target score s_t(x_t|x_0) = - noise / sqrt(1 - alpha_bar)
            target = - noise / sqrt_1m_alpha_bars[t].unsqueeze(1)
            
            # Model prediction s_theta(x_t, t)
            pred = score_net(x_t, t, T)
            
            # Loss is MSE between prediction and target score (Denoising Objective)
            loss = torch.mean((pred - target) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")
            
    # Save Model
    torch.save(score_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()