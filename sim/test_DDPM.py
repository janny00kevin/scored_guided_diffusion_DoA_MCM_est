import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from models.score_net import ScoreNet

# =============================
# 1. Configuration
# =============================
if not torch.cuda.is_available():
    print("Error: GPU required for batch processing.")
    exit()

device = torch.device("cuda")
TEST_DATA_PATH = "dataset/test_data_all_snr.pt"
MODEL_PATH = "weights/DDPM_ep1000_lr1e-03_t50_bmin1e-04_bmax0.02.pth"
OUTPUT_PATH = "test_results/test_results_DDPM.pt"

# Diffusion Params
T = 50
BETA_MIN = 1e-4
BETA_MAX = 0.02
GUIDANCE_LAMBDA = 0.8
NUM_STEPS = 50

# Precompute schedules
betas = torch.linspace(BETA_MIN, BETA_MAX, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_1m_alpha_bars = torch.sqrt(1.0 - alpha_bars)

# =============================
# 2. Batched SDE Sampler
# =============================
def complex_to_real(x):
    return torch.cat([x.real, x.imag], dim=-1)

def real_to_complex(x):
    N = x.shape[-1] // 2
    return x[..., :N] + 1j * x[..., N:]

def batch_euler_maruyama_sampler(Y_batch, score_net, snr_db):
    """
    Y_batch: (Batch, N, L) Complex
    Returns: X_est (Batch, N, L) Complex
    """
    B, N, L = Y_batch.shape
    dt = 1.0 / T
    
    # 1. Flatten Batch and Snapshots -> (B*L, N)
    # This allows us to feed everything into ScoreNet at once
    Y_reshaped = Y_batch.permute(0, 2, 1).reshape(B * L, N) # (B*L, N)
    y_real = complex_to_real(Y_reshaped) # (B*L, 2N)
    
    # Noise variance for guidance
    sigma_pwr = 10 ** (-snr_db / 10.0)
    sigma_y2 = (sigma_pwr / 2.0)

    # 2. Initialize x_T ~ N(0, I)
    x_t = torch.randn_like(y_real, device=device)

    # 3. Reverse Diffusion Loop
    # This loop runs 50 times, but processes B*L samples in parallel
    for i in reversed(range(NUM_STEPS)):
        t_idx = torch.full((B * L,), i, device=device, dtype=torch.long)
        
        beta = betas[i]
        a_bar = alpha_bars[i]
        sqrt_ab = sqrt_alpha_bars[i]
        
        # Predict Score
        score_pred = score_net(x_t, t_idx, T)
        
        # Physics Guidance
        # x0_hat = (x_t + (1 - a_bar) * score) / sqrt_ab
        x0_hat = (x_t + (1.0 - a_bar) * score_pred) / (sqrt_ab + 1e-12)
        
        # Likelihood gradient
        grad_likelihood = (y_real - x0_hat) / (sigma_y2 + 1e-8)
        grad_xt = grad_likelihood / (sqrt_ab + 1e-8)
        
        total_score = score_pred + GUIDANCE_LAMBDA * grad_xt
        
        # Euler Step
        drift = -0.5 * beta * x_t - beta * total_score
        diffusion = torch.sqrt(beta * dt) * torch.randn_like(x_t)
        
        if i > 0:
            x_t = x_t + drift * dt + diffusion
        else:
            x_t = x_t + drift * dt

    # 4. Reshape back to (B, N, L)
    x0_complex = real_to_complex(x_t) # (B*L, N)
    X_est = x0_complex.reshape(B, L, N).permute(0, 2, 1) # (B, N, L)
    return X_est

# =============================
# 3. Batched EM Solver (Copied & Adapted)
# =============================
# We include these helpers here to avoid import issues and ensure consistency
def batch_steering_vector(N, theta, device):
    B, P = theta.shape
    theta_rad = theta * (np.pi / 180.0)
    k = 2.0 * np.pi * 0.5 * torch.sin(theta_rad)
    n = torch.arange(0, N, device=device).view(1, N, 1)
    phase = -1j * (n * k.unsqueeze(1))
    return torch.exp(phase)

def batch_build_M_from_c(c_param, N, device):
    B, K = c_param.shape
    M = torch.zeros(B, N, N, dtype=torch.complex64, device=device)
    for k in range(K):
        val = c_param[:, k]
        row_idx = torch.arange(0, N - k, device=device)
        col_idx = torch.arange(k, N, device=device)
        M[:, row_idx, col_idx] = val.view(B, 1)
        if k > 0:
            M[:, col_idx, row_idx] = val.conj().view(B, 1)
    return M

def batch_run_em_solver(Y_batch, N, P, toeplitz_K=5, num_outer=5, num_inner=50):
    B = Y_batch.shape[0]
    
    # Init Theta (Music)
    Ry = torch.matmul(Y_batch, Y_batch.conj().transpose(1, 2)) / Y_batch.shape[2]
    angles_grid = torch.linspace(-90, 90, 181, device=device)
    A_grid = batch_steering_vector(N, angles_grid.unsqueeze(0), device).squeeze(0)
    Ry_inv = torch.linalg.pinv(Ry)
    prod = torch.matmul(Ry_inv, A_grid.unsqueeze(0)) 
    denom = torch.sum(A_grid.unsqueeze(0).conj() * prod, dim=1).real
    _, idx = torch.topk(1.0/(denom+1e-12), P, dim=1)
    theta_est = angles_grid[idx].detach().requires_grad_(True)
    
    # Init M (c_param)
    c_init = torch.zeros(B, toeplitz_K, dtype=torch.complex64, device=device)
    c_init[:, 0] = 1.0
    for k in range(1, toeplitz_K): c_init[:, k] = 0.05 * (0.5**k)
    c_param = c_init.detach().requires_grad_(True)
    
    Ry_norm = torch.mean(torch.abs(Ry)**2, dim=(1,2))

    for outer in range(num_outer):
        # Update Theta
        opt_theta = optim.Adam([theta_est], lr=0.05)
        for _ in range(num_inner):
            opt_theta.zero_grad()
            A = batch_steering_vector(N, theta_est, device)
            M_raw = batch_build_M_from_c(c_param, N, device)
            # Normalize M
            m00 = M_raw[:,0,0].view(B,1,1)
            M = (M_raw / (m00/m00.abs())) / m00.real
            
            T1 = M @ A
            R_model = T1 @ T1.conj().transpose(1,2)
            loss = torch.mean(torch.mean(torch.abs(Ry - R_model)**2, dim=(1,2))/Ry_norm)
            loss.backward()
            opt_theta.step()
            with torch.no_grad(): theta_est.clamp_(-90, 90)
            
        # Update M
        opt_M = optim.Adam([c_param], lr=0.01)
        for _ in range(num_inner):
            opt_M.zero_grad()
            A = batch_steering_vector(N, theta_est, device)
            M_raw = batch_build_M_from_c(c_param, N, device)
            m00 = M_raw[:,0,0].view(B,1,1)
            M = (M_raw / (m00/m00.abs())) / m00.real
            
            T1 = M @ A
            R_model = T1 @ T1.conj().transpose(1,2)
            loss = torch.mean(torch.mean(torch.abs(Ry - R_model)**2, dim=(1,2))/Ry_norm)
            loss.backward()
            opt_M.step()
            
    # Finalize
    theta_est, _ = torch.sort(theta_est, dim=1)
    M_raw = batch_build_M_from_c(c_param, N, device)
    m00 = M_raw[:,0,0].view(B,1,1)
    M_final = (M_raw / (m00/m00.abs())) / m00.real
    return theta_est, M_final

# =============================
# 4. Main Execution
# =============================
def run_benchmark():
    if not os.path.exists(TEST_DATA_PATH):
        print("Data not found.")
        return
    
    # Load ScoreNet
    score_net = ScoreNet(dim=2*16).to(device)
    score_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    score_net.eval()
    
    # Load Dataset
    full_dataset = torch.load(TEST_DATA_PATH)
    snr_levels = sorted(full_dataset.keys())
    
    final_results = {
        "snr_levels": snr_levels,
        "doa_nmse_avg": [],
        "mcm_nmse_avg": []
    }
    
    print("Starting Method 2 (Batch GPU) Benchmark...")
    
    for snr in snr_levels:
        samples = full_dataset[snr]
        print(f"Processing SNR {snr}dB ({len(samples)} samples)...")
        
        # 1. Prepare Batch Data
        Y_batch = torch.stack([s['Y'] for s in samples]).to(device) # (B, N, L)
        theta_true = torch.stack([s['theta_true'] for s in samples]).to(device)
        M_true = torch.stack([s['M_true'] for s in samples]).to(device)
        
        # 2. Run SDE Sampling (Batch)
        with torch.no_grad():
            X_est = batch_euler_maruyama_sampler(Y_batch, score_net, snr)
        
        # 3. Run EM Estimation (Batch)
        # Pass the cleaned signal X_est as if it were the observation
        theta_est, M_est = batch_run_em_solver(X_est, 16, 3)
        
        # 4. Metrics
        theta_true_sorted, _ = torch.sort(theta_true, dim=1)
        
        err_doa = torch.norm(theta_est - theta_true_sorted, dim=1) / torch.norm(theta_true_sorted, dim=1)
        err_mcm = torch.norm(M_est - M_true, dim=(1,2)) / torch.norm(M_true, dim=(1,2))
        
        avg_doa_db = 20 * torch.log10(err_doa).mean().item()
        avg_mcm_db = 20 * torch.log10(err_mcm).mean().item()
        
        final_results["doa_nmse_avg"].append(avg_doa_db)
        final_results["mcm_nmse_avg"].append(avg_mcm_db)
        print(f"  -> Avg DOA: {avg_doa_db:.2f} dB | Avg MCM: {avg_mcm_db:.2f} dB")
        
    torch.save(final_results, OUTPUT_PATH)
    print(f"Done. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_benchmark()