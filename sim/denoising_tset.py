import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
# 引入兩個不同的模型定義
from models.epsnet_mlp import EpsNetMLP      # 給 Dynamic T=1000 用
from models.epsnet_unet1d import EpsNetUNet1D # 給 Fixed T=50 用

torch.manual_seed(0)

# =============================
# 1. Configuration & Global Setup
# =============================
if not torch.cuda.is_available():
    print("Error: GPU required.")
    exit()

device = torch.device("cuda") 
TEST_DATA_PATH = "dataset/test_data_all_snr.pt"

# --- Model 1: Dynamic (T=1000, Normalized) ---
MODEL_PATH_1000 = "weights/DDIM_ep50_lr1e-03_t1000_bmax2e-02_nmlz.pth"
T_1000 = 1000

# --- Model 2: Fixed (T=50, No Norm) ---
MODEL_PATH_50 = "weights/DDIM_ep50_lr1e-03_t50_bmax_2e-02.pth"
T_50 = 50

GUIDANCE_LAMBDA_FIX = 0.4  # 全域 Lambda 設定
GUIDANCE_LAMBDA_DY = 1
NUM_TEST_SAMPLES = 600

# =============================
# 2. Precompute Schedules (Two Sets)
# =============================
def get_schedule(T, device):
    beta_min = 1e-4
    beta_max = 0.02
    betas = torch.linspace(beta_min, beta_max, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars

# 分別建立兩套 Schedule
alpha_bars_1000 = get_schedule(T_1000, device)
alpha_bars_50   = get_schedule(T_50, device)

# =============================
# 3. Helpers
# =============================
def complex_to_real(x):
    return torch.cat([x.real, x.imag], dim=-1)

def real_to_complex(x):
    N = x.shape[-1] // 2
    return x[..., :N] + 1j * x[..., N:]

# =============================
# 4. Samplers
# =============================

def batch_ddim_sampler_dynamic_norm_T1000(Y_batch, eps_net, snr_db, data_mean, data_std):
    """
    [Dynamic Lambda Version]
    - T = 1000
    - With Normalization (data_mean, data_std)
    - With Chain Rule (grad * data_std)
    - With Time Decay (lambda * sqrt(1-alpha))
    """
    B, N, L = Y_batch.shape
    Y_reshaped = Y_batch.permute(0, 2, 1).reshape(B * L, N)
    y_real_phys = complex_to_real(Y_reshaped) # 物理空間
    
    sigma_pwr = 10 ** (-snr_db / 10.0)
    sigma_y2 = (sigma_pwr / 2.0)

    # 1. Init x_T (Normalized Space -> randn)
    x_t = torch.randn_like(y_real_phys, device=device)
    
    # Ensure shapes for broadcasting
    data_mean = data_mean.view(1, 1)
    data_std = data_std.view(1, 1)

    t_seq = torch.linspace(0, 1000-1, 50).long().flip(0).tolist() # 999 -> 0
    
    for i in range(len(t_seq)):
        t_cur = t_seq[i]
        t_prev = t_seq[i+1] if i < len(t_seq) - 1 else -1
        t_batch = torch.full((B * L,), t_cur, device=device, dtype=torch.long)
        
        with torch.no_grad():
            eps_pred = eps_net(x_t, t_batch)
        
        # 使用 T=1000 的 Schedule
        a_bar_cur = alpha_bars_1000[t_cur]
        sqrt_a_cur = torch.sqrt(a_bar_cur)
        sqrt_1m_a_cur = torch.sqrt(1.0 - a_bar_cur)
        
        # x0_hat (Normalized)
        x0_hat_norm = (x_t - sqrt_1m_a_cur * eps_pred) / (sqrt_a_cur + 1e-12)
        
        # === Dynamic Guidance (Normalized Logic) ===
        if GUIDANCE_LAMBDA_DY > 0:
            # A. 轉回物理空間算殘差
            x0_hat_phys = x0_hat_norm * data_std + data_mean
            
            # B. 物理 Gradient (含保護)
            grad_phys = (y_real_phys - x0_hat_phys) / max(sigma_y2, 1)
            
            # C. Chain Rule: 轉回 Normalized Gradient
            grad_norm = grad_phys * data_std
            
            # D. 動態衰減 Lambda
            # 注意：T=1000且有Normalize時，通常需要較大的Multiplier來補償std的縮放，
            # 這裡維持原本邏輯，若效果太弱可嘗試 GUIDANCE_LAMBDA * 100 * ...
            effective_lambda = GUIDANCE_LAMBDA_DY * sqrt_1m_a_cur
            
            x0_hat_guided_norm = x0_hat_norm + effective_lambda * grad_norm
        else:
            x0_hat_guided_norm = x0_hat_norm

        # DDIM Update
        if t_prev >= 0:
            a_bar_prev = alpha_bars_1000[t_prev]
            sqrt_a_prev = torch.sqrt(a_bar_prev)
            sqrt_1m_a_prev = torch.sqrt(1.0 - a_bar_prev)
            
            eps_guided = (x_t - sqrt_a_cur * x0_hat_guided_norm) / (sqrt_1m_a_cur + 1e-12)
            x_t = sqrt_a_prev * x0_hat_guided_norm + sqrt_1m_a_prev * eps_guided
        else:
            x_t = x0_hat_guided_norm

    # 最後還原回物理空間
    x0_final_phys = x_t * data_std + data_mean
    x0_complex = real_to_complex(x0_final_phys)
    X_est = x0_complex.reshape(B, L, N).permute(0, 2, 1)
    return X_est


def batch_ddim_sampler_fixed_nonorm_T50(Y_batch, eps_net, snr_db):
    """
    [Fixed Lambda Version]
    - T = 50
    - NO Normalization (Input/Output are physical)
    - Constant Lambda (No time decay)
    """
    B, N, L = Y_batch.shape
    Y_reshaped = Y_batch.permute(0, 2, 1).reshape(B * L, N)
    y_real = complex_to_real(Y_reshaped) 
    
    sigma_pwr = 10 ** (-snr_db / 10.0)
    sigma_y2 = (sigma_pwr / 2.0)

    # Init x_T (Physical Space -> randn)
    x_t = torch.randn_like(y_real, device=device)
    
    t_seq = list(reversed(range(T_50))) # 49 -> 0
    
    for i in range(len(t_seq)):
        t_cur = t_seq[i]
        t_prev = t_seq[i+1] if i < len(t_seq) - 1 else -1
        t_batch = torch.full((B * L,), t_cur, device=device, dtype=torch.long)
        
        with torch.no_grad():
            eps_pred = eps_net(x_t, t_batch)
        
        # 使用 T=50 的 Schedule
        a_bar_cur = alpha_bars_50[t_cur]
        sqrt_a_cur = torch.sqrt(a_bar_cur)
        sqrt_1m_a_cur = torch.sqrt(1.0 - a_bar_cur)
        
        x0_hat = (x_t - sqrt_1m_a_cur * eps_pred) / (sqrt_a_cur + 1e-12)
        
        # === Fixed Guidance (No Norm Logic) ===
        if GUIDANCE_LAMBDA_FIX > 0:
            # 直接計算 Gradient
            grad_x0 = (y_real - x0_hat) / (sigma_y2 + 1e-8)
            
            # 直接更新 (Constant)
            x0_hat_guided = x0_hat + GUIDANCE_LAMBDA_FIX * grad_x0
        else:
            x0_hat_guided = x0_hat

        # DDIM Update
        if t_prev >= 0:
            a_bar_prev = alpha_bars_50[t_prev]
            sqrt_a_prev = torch.sqrt(a_bar_prev)
            sqrt_1m_a_prev = torch.sqrt(1.0 - a_bar_prev)
            
            eps_guided = (x_t - sqrt_a_cur * x0_hat_guided) / (sqrt_1m_a_cur + 1e-12)
            x_t = sqrt_a_prev * x0_hat_guided + sqrt_1m_a_prev * eps_guided
        else:
            x_t = x0_hat_guided

    x0_complex = real_to_complex(x_t)
    X_est = x0_complex.reshape(B, L, N).permute(0, 2, 1)
    return X_est

# =============================
# 5. Evaluation Loop
# =============================
def run_denoising_check():
    print("Loading Models...")
    
    # --- Load Model 1 (T=1000, Norm, MLP) ---
    print(f"Loading Dynamic Model (T={T_1000}): {MODEL_PATH_1000}")
    checkpoint_1000 = torch.load(MODEL_PATH_1000, map_location=device)
    net_1000 = EpsNetMLP(dim=2*16, hidden=1024, time_emb_dim=128).to(device)
    net_1000.load_state_dict(checkpoint_1000['model_state_dict'])
    net_1000.eval()
    
    # 讀取 Normalization 參數
    data_mean = checkpoint_1000['data_mean'].to(device)
    data_std = checkpoint_1000['data_std'].to(device)
    
    # --- Load Model 2 (T=50, No Norm, UNet1D) ---
    print(f"Loading Fixed Model (T={T_50}): {MODEL_PATH_50}")
    net_50 = EpsNetUNet1D(dim=2*16).to(device)
    checkpoint_50 = torch.load(MODEL_PATH_50, map_location=device)
    # 處理可能的 dict 結構
    if isinstance(checkpoint_50, dict) and 'model_state_dict' in checkpoint_50:
        net_50.load_state_dict(checkpoint_50['model_state_dict'])
    else:
        net_50.load_state_dict(checkpoint_50)
    net_50.eval()

    POWER_OFFSET_DB = 10 * np.log10(3.0)
    full_dataset = torch.load(TEST_DATA_PATH)
    snr_levels = sorted(full_dataset.keys())
    
    snr_list = []
    nmse_in_list = []
    nmse_dyn_list = [] 
    nmse_fix_list = [] 

    print(f"\n{'SNR (dB)':<10} | {'Input NMSE':<12} | {'Dynamic':<12} | {'Fixed':<12}")
    print(f"{'':<10} | {'':<12} | {'(T=1000,Nmlz)':<12} | {'(T=50,Raw)':<12}")
    print("-" * 60)

    from training_all_snapshots_same_time_PINN_MCM_proj_steering_vec.data.generator import generate_snapshot_sample

    for snr in snr_levels:
        # Generate Data
        X_true_list = []
        Y_obs_list = []
        for _ in range(NUM_TEST_SAMPLES):
             X, Y, _, _, _ = generate_snapshot_sample(16, 3, 128, snr, device, randomize=True, use_toeplitz=True)
             X_true_list.append(X)
             Y_obs_list.append(Y)
        
        X_true = torch.stack(X_true_list)
        Y_batch = torch.stack(Y_obs_list)
        
        # 1. Input NMSE
        diff_in = Y_batch - X_true
        norm_diff_in = torch.norm(diff_in, dim=(1,2)) ** 2
        norm_X = torch.norm(X_true, dim=(1,2)) ** 2
        nmse_in_db = 10 * np.log10(torch.mean(norm_diff_in / norm_X).item()) + POWER_OFFSET_DB
        
        # 2. Dynamic Sampler (Model 1000)
        with torch.no_grad():
            X_dyn = batch_ddim_sampler_dynamic_norm_T1000(Y_batch, net_1000, snr, data_mean, data_std)
        
        diff_dyn = X_dyn - X_true
        norm_diff_dyn = torch.norm(diff_dyn, dim=(1,2)) ** 2
        nmse_dyn_db = 10 * np.log10(torch.mean(norm_diff_dyn / norm_X).item()) + POWER_OFFSET_DB
        
        # 3. Fixed Sampler (Model 50)
        with torch.no_grad():
            X_fix = batch_ddim_sampler_fixed_nonorm_T50(Y_batch, net_50, snr)
            
        diff_fix = X_fix - X_true
        norm_diff_fix = torch.norm(diff_fix, dim=(1,2)) ** 2
        nmse_fix_db = 10 * np.log10(torch.mean(norm_diff_fix / norm_X).item()) + POWER_OFFSET_DB
        
        print(f"{snr:<10} | {nmse_in_db:<12.2f} | {nmse_dyn_db:<12.2f} | {nmse_fix_db:<12.2f}")
        
        snr_list.append(snr)
        nmse_in_list.append(nmse_in_db)
        nmse_dyn_list.append(nmse_dyn_db)
        nmse_fix_list.append(nmse_fix_db)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(snr_list, nmse_in_list, 'k--', marker='o', label='Input NMSE')
    plt.plot(snr_list, nmse_fix_list, 'r-^', linewidth=2, label=f'Fixed (T=50, Raw)')
    plt.plot(snr_list, nmse_dyn_list, 'g-s', linewidth=2, label=f'Dynamic (T=1000, Nmlz)')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title(f'Denoising Comparison: Fixed T50 vs Dynamic T1000')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('test_results/denoising_result_comparison.png')
    print("\nPlot saved to test_results/denoising_result_comparison.png")

    mat_path = 'test_results/denoising_data_comparison.mat'
    sio.savemat(mat_path, {
        "snr_axis": np.array(snr_list),
        "nmse_in": np.array(nmse_in_list),
        "nmse_out_dynamic": np.array(nmse_dyn_list),
        "nmse_out_fixed": np.array(nmse_fix_list)
    })
    print(f"Data saved to {mat_path}")

if __name__ == "__main__":
    run_denoising_check()