import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from models.epsnet_mlp import EpsNetMLP

# =============================
# 1. Configuration
# =============================
if not torch.cuda.is_available():
    print("Error: GPU required.")
    exit()

device = torch.device("cuda:1")
TEST_DATA_PATH = "dataset/test_data_all_snr.pt"
MODEL_PATH = "weights/DDIM_ep50_lr1e-03_t1000_bmax2e-02_nmlz.pth"

# DDIM Params
T = 1000
GUIDANCE_LAMBDA = 1   #################
NUM_STEPS = 50 
BATCH_SIZE = 3000
NUM_TEST_SAMPLES = 3000
OUTPUT_PATH = f"test_results/test_results_DDIM_lamb{GUIDANCE_LAMBDA:.0e}_bmax2e-02_nmlz_.pt"  #######
# Precompute schedules
beta_min = 1e-4
beta_max = 0.02
betas = torch.linspace(beta_min, beta_max, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_1m_alpha_bars = torch.sqrt(1.0 - alpha_bars)

# =============================
# 2. Batched DDIM Sampler
# =============================
def complex_to_real(x):
    return torch.cat([x.real, x.imag], dim=-1)

def real_to_complex(x):
    N = x.shape[-1] // 2
    return x[..., :N] + 1j * x[..., N:]

def alpha_bar_of_t(t_idx):
    # Helper to get alpha_bar at integer index t
    return alpha_bars[t_idx]

def batch_ddim_sampler(Y_batch, eps_net, snr_db, data_mean, data_std, num_steps=50, T=1000):
    """
    Note: Added data_mean and data_std parameters
    """
    B, N, L = Y_batch.shape
    
    # 1. Process observations Y (unchanged, keep physical values)
    Y_reshaped = Y_batch.permute(0, 2, 1).reshape(B * L, N)
    y_real_phys = complex_to_real(Y_reshaped) # Y in real physical space
    
    # Calculate noise variance in physical space (used for Guidance)
    sigma_pwr = 10 ** (-snr_db / 10.0)
    sigma_y2 = (sigma_pwr / 2.0)

    # 2. Initialize x_T (in normalized space)
    # Since the input during training is N(0,1), we can use randn directly here 
    # without multiplying by data_std
    x_t = torch.randn_like(y_real_phys, device=device)
    
    # Ensure mean/std have the correct shape (1, 1) for broadcasting
    data_mean = data_mean.view(1, 1)
    data_std = data_std.view(1, 1)

    t_seq = torch.linspace(0, T-1, num_steps).long().flip(0).tolist()
    
    for i in range(len(t_seq)):
        t_cur = t_seq[i]
        t_prev = t_seq[i+1] if i < len(t_seq) - 1 else -1
        
        t_batch = torch.full((B * L,), t_cur, device=device, dtype=torch.long)
        
        # 3. Predict Eps (Model operates in normalized space)
        with torch.no_grad():
            eps_pred = eps_net(x_t, t_batch)
        
        # 4. Calculate x0_hat (This is the estimate in normalized space)
        a_bar_cur = alpha_bars[t_cur]
        sqrt_a_cur = torch.sqrt(a_bar_cur)
        sqrt_1m_a_cur = torch.sqrt(1.0 - a_bar_cur)
        
        x0_hat_norm = (x_t - sqrt_1m_a_cur * eps_pred) / (sqrt_a_cur + 1e-12)
        
        # === [Key] Physical Guidance ===
        if GUIDANCE_LAMBDA > 0:
            # A. Revert to physical space
            x0_hat_phys = x0_hat_norm * data_std + data_mean
            
            # B. Calculate Gradient in physical space
            # grad = (y - x) / sigma^2
            # Suggested adding denominator protection here (as discussed previously)
            grad_phys = (y_real_phys - x0_hat_phys) / max(sigma_y2, 1)
            
            # C. [Most Important] Convert Gradient back to normalized space (Chain Rule)
            # Since x_phys = x_norm * std + mean
            # Thus dx_phys / dx_norm = std
            # Gradient propagation: grad_norm = grad_phys * std
            grad_norm = grad_phys * data_std
            
            # D. Update x0_hat (in normalized space)
            x0_hat_guided_norm = x0_hat_norm + GUIDANCE_LAMBDA * sqrt_1m_a_cur * grad_norm
        else:
            x0_hat_guided_norm = x0_hat_norm

        # 5. DDIM Update (performed in normalized space)
        if t_prev >= 0:
            a_bar_prev = alpha_bars[t_prev]
            sqrt_a_prev = torch.sqrt(a_bar_prev)
            sqrt_1m_a_prev = torch.sqrt(1.0 - a_bar_prev)
            
            # Re-calculate eps corresponding to guided x0
            eps_guided = (x_t - sqrt_a_cur * x0_hat_guided_norm) / (sqrt_1m_a_cur + 1e-12)
            
            x_t = sqrt_a_prev * x0_hat_guided_norm + sqrt_1m_a_prev * eps_guided
        else:
            x_t = x0_hat_guided_norm

    # 6. Revert to physical space before final output
    x0_final_phys = x_t * data_std + data_mean
    
    # Convert back to complex
    x0_complex = real_to_complex(x0_final_phys)
    X_est = x0_complex.reshape(B, L, N).permute(0, 2, 1)
    return X_est

# =============================
# 3. Batched EM Solver (Reused)
# =============================
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
    
    # Init M
    c_init = torch.zeros(B, toeplitz_K, dtype=torch.complex64, device=device)
    c_init[:, 0] = 1.0
    for k in range(1, toeplitz_K): c_init[:, k] = 0.05 * (0.5**k)
    c_param = c_init.detach().requires_grad_(True)
    
    Ry_norm = torch.mean(torch.abs(Ry)**2, dim=(1,2))

    for outer in range(num_outer):
        opt_theta = optim.Adam([theta_est], lr=0.05)
        for _ in range(num_inner):
            opt_theta.zero_grad()
            A = batch_steering_vector(N, theta_est, device)
            M_raw = batch_build_M_from_c(c_param, N, device)
            m00 = M_raw[:,0,0].view(B,1,1)
            M = (M_raw / (m00/m00.abs())) / m00.real
            T1 = M @ A
            R_model = T1 @ T1.conj().transpose(1,2)
            loss = torch.mean(torch.mean(torch.abs(Ry - R_model)**2, dim=(1,2))/Ry_norm)
            loss.backward()
            opt_theta.step()
            with torch.no_grad(): theta_est.clamp_(-90, 90)
            
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
    
    # Load EpsNet
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    eps_net = EpsNetMLP(dim=2*16, hidden=1024, time_emb_dim=128).to(device)
    eps_net.load_state_dict(checkpoint['model_state_dict'])
    eps_net.eval()
    
    data_mean = checkpoint['data_mean'].to(device)
    data_std = checkpoint['data_std'].to(device)

    full_dataset = torch.load(TEST_DATA_PATH)
    snr_levels = sorted(full_dataset.keys())
    
    final_results = {
        "snr_levels": snr_levels,
        "doa_nmse_avg": [],
        "mcm_nmse_avg": []
    }
    
    print(f"Starting Method 3 Benchmark (Batch Size: {BATCH_SIZE})...")
    
    for snr in snr_levels:
        samples = full_dataset[snr][:NUM_TEST_SAMPLES]
        num_samples = len(samples)
        print(f"Processing SNR {snr}dB ({num_samples} samples)...")
        
        # Temporarily store all errors for this SNR level
        all_doa_errs = []
        all_mcm_errs = []
        
        # === [Key Modification] Mini-Batch Loop ===
        # Process data in chunks of BATCH_SIZE to manage GPU memory
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc=f"SNR {snr}dB"):
            batch_samples = samples[i : i + BATCH_SIZE]
            
            # 1. Prepare Batch Data
            Y_batch = torch.stack([s['Y'] for s in batch_samples]).to(device) # (B, N, L)
            theta_true = torch.stack([s['theta_true'] for s in batch_samples]).to(device)
            M_true = torch.stack([s['M_true'] for s in batch_samples]).to(device)
            
            # 2. DDIM Denoising
            with torch.no_grad():
                X_est = batch_ddim_sampler(Y_batch, eps_net, snr, data_mean, data_std)
            
            # 3. EM Estimation
            theta_est, M_est = batch_run_em_solver(X_est, 16, 3, toeplitz_K=4)
            
            # 4. Metrics
            theta_true_sorted, _ = torch.sort(theta_true, dim=1)
            
            # Calculate errors for this batch
            err_doa = torch.norm(theta_est - theta_true_sorted, dim=1) / torch.norm(theta_true_sorted, dim=1)
            err_mcm = torch.norm(M_est - M_true, dim=(1,2)) / torch.norm(M_true, dim=(1,2))
            
            # Convert to dB and append to the temporary list
            # Note: Convert to CPU list immediately to avoid holding GPU memory
            all_doa_errs.extend((20 * torch.log10(err_doa)).tolist())
            all_mcm_errs.extend((20 * torch.log10(err_mcm)).tolist())
            
            # Clear GPU memory (optional, but helpful when memory is tight)
            del Y_batch, theta_true, M_true, X_est, theta_est, M_est
            torch.cuda.empty_cache()

        # === After loop, calculate average metrics for this SNR ===
        # Filter out potential -inf values (if error is exactly 0)
        clean_doa = [e for e in all_doa_errs if e > -120]
        clean_mcm = [e for e in all_mcm_errs if e > -120]
        
        avg_doa_db = np.mean(clean_doa) if clean_doa else -100.0
        avg_mcm_db = np.mean(clean_mcm) if clean_mcm else -100.0
        
        final_results["doa_nmse_avg"].append(avg_doa_db)
        final_results["mcm_nmse_avg"].append(avg_mcm_db)
        print(f"  -> Avg DOA: {avg_doa_db:.2f} dB | Avg MCM: {avg_mcm_db:.2f} dB")
        
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
    torch.save(final_results, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_benchmark()