import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

from em.stable_em_batch import steering_vector_batch

# =============================
# 1. Configuration
# =============================
# 強制使用 GPU，如果沒有則報錯
if not torch.cuda.is_available():
    print("Error: This script requires a GPU to run efficiently.")
    exit()

device = torch.device("cuda")
torch.set_default_dtype(torch.float32)

# =============================
# 2. Batched Helper Functions
# =============================
def batch_steering_vector(N, theta, device):
    """
    Generates steering vectors for a batch of angles.
    theta: (Batch, P)
    Returns: (Batch, N, P)
    """
    B, P = theta.shape
    theta_rad = theta * (np.pi / 180.0)
    d = 0.5
    k = 2.0 * np.pi * d * torch.sin(theta_rad) # (B, P)
    n = torch.arange(0, N, dtype=torch.float32, device=device).view(1, N, 1) # (1, N, 1)
    
    # Phase: -1j * n * k
    # (1, N, 1) * (B, 1, P) -> (B, N, P) via broadcasting
    phase = -1j * (n * k.unsqueeze(1)) 
    return torch.exp(phase)

def batch_build_M_from_c(c_param, N, device):
    """
    Constructs a batch of Toeplitz matrices.
    c_param: (Batch, K) - The first K parameters of the first column
    Returns: (Batch, N, N)
    """
    B, K = c_param.shape
    M = torch.zeros(B, N, N, dtype=torch.complex64, device=device)
    
    # Vectorized diagonal filling
    # Loop over K (bandwidth) is small (e.g., 5), loop over Batch is handled by tensor ops
    for k in range(K):
        val = c_param[:, k] # (B,)
        
        # Create a diagonal mask or use advanced indexing could be faster, 
        # but creating identity matrices is memory intensive for large B.
        # We use a loop over K which is efficient enough.
        
        # Construct diagonal matrices for the batch
        # This part is slightly tricky to vectorize perfectly without high memory, 
        # but since K is small, we iterate K.
        
        # Current diagonal: k
        # We need to fill M[:, i, i+k] = val and M[:, i+k, i] = conj(val)
        
        # Create indices
        row_idx = torch.arange(0, N - k, device=device)
        col_idx = torch.arange(k, N, device=device)
        
        # Assign values using advanced indexing
        # val.view(B, 1) expands to (B, N-k)
        M[:, row_idx, col_idx] = val.view(B, 1)
        
        if k > 0:
            M[:, col_idx, row_idx] = val.conj().view(B, 1)
            
    return M

def batch_compute_covariance(Y):
    """
    Y: (Batch, N, L)
    Returns: (Batch, N, N)
    """
    B, N, L = Y.shape
    # R = Y @ Y^H / L
    # (B, N, L) @ (B, L, N) -> (B, N, N)
    return torch.matmul(Y, Y.conj().transpose(1, 2)) / L

def batch_music_initialization(Y, P, N, device):
    """
    Batched MUSIC algorithm.
    Y: (Batch, N, L)
    Returns: (Batch, P) - Initial guesses for theta
    """
    B = Y.shape[0]
    Ry = batch_compute_covariance(Y) # (B, N, N)
    
    # Grid search
    angles_grid = torch.linspace(-90, 90, 181, device=device) # (181,)
    A_grid = batch_steering_vector(N, angles_grid.unsqueeze(0), device).squeeze(0) # (N, 181)
    
    # Compute inverse of Ry
    # Use pinv for stability or solve
    # (Batch, N, N)
    Ry_inv = torch.linalg.pinv(Ry) 
    
    # Compute spectrum: P(theta) = 1 / (a^H Ry^-1 a)
    # We want to do this for all B and all 181 angles at once.
    
    # A_grid: (N, 181)
    # Ry_inv: (B, N, N)
    # Target: (B, 181)
    
    # Quadratic form: x^H A x can be done via:
    # (Ry_inv @ A_grid) -> (B, N, 181)
    # Then elementwise multiply with A_grid.conj() and sum over N
    
    # 1. Ry_inv @ A_grid
    # We treat A_grid as (1, N, 181) and broadcast
    prod = torch.matmul(Ry_inv, A_grid.unsqueeze(0)) # (B, N, 181)
    
    # 2. a^H @ prod
    # A_grid_conj: (1, N, 181)
    # term = sum(conj(A) * prod, dim=1)
    denom = torch.sum(A_grid.unsqueeze(0).conj() * prod, dim=1).real # (B, 181)
    
    spectrum = 1.0 / (denom + 1e-12)
    
    # Find peaks (Top K)
    _, idx = torch.topk(spectrum, P, dim=1) # (B, P) indices
    
    # Gather angles
    # angles_grid[idx]
    theta_init = angles_grid[idx] # (B, P)
    
    return theta_init

def build_M_from_toeplitz_params_batch(c_param_batch, N):
    """
    Batch construction of Toeplitz matrices.
    Args:
        c_param_batch: (B, K) - first column parameters
    Returns:
        M: (B, N, N)
    """
    B, K = c_param_batch.shape
    device = c_param_batch.device
    
    M = torch.eye(N, dtype=torch.complex64, device=device).unsqueeze(0).repeat(B, 1, 1)
    
    # c_param_batch[:, 0] is diagonal, usually handled or normalized separately
    # logic similar to original:
    for k in range(1, min(K, N)):
        # val: (B,)
        val = c_param_batch[:, k]
        # Construct diagonals efficiently using diag_embed logic or masking
        # For batch, simple loop over k adding to diagonals is fine
        
        # Create a diagonal mask for offset k
        # We can construct the diagonal matrix for this k
        # (B, N-k)
        ones = torch.ones(N - k, dtype=torch.complex64, device=device)
        diag_vals = val.unsqueeze(1) * ones.unsqueeze(0) # (B, N-k)
        
        M += torch.diag_embed(diag_vals, offset=k) + torch.diag_embed(torch.conj(diag_vals), offset=-k)
        
    return M

# =============================
# 3. Batched Solver
# =============================
def run_batch_solver(x0_batch, N, P, num_outer=2, num_inner=50, toeplitz_K=5, lr_theta=1e-2, lr_M=1e-2):
    """
    Runs the alternating estimation on the GPU for the entire batch.
    """
    B = x0_batch.shape[0]
    L = x0_batch.shape[2]

    
    R_y = torch.matmul(x0_batch, x0_batch.conj().transpose(1, 2)) / L
    Ry_norm = torch.mean(torch.abs(R_y)**2, dim=(1, 2)) # (B,)

    
    # 1. Initialization
    print("  -> Running Batched MUSIC Init...")
    theta_est = batch_music_initialization(x0_batch, P, N, device)
    theta_est = theta_est.detach().requires_grad_(True)
    
    # M initialization (Toeplitz)
    if toeplitz_K is not None:
        c_init = torch.zeros(B, toeplitz_K, dtype=torch.complex64, device=device)
        c_init[:, 0] = 1.0 + 0.0j
        for k in range(1, toeplitz_K):
            c_init[:, k] = 0.05 * (0.5**k) + 0j
        c_param = c_init.clone().detach().requires_grad_(True)
        optimizer_M = optim.Adam([c_param], lr=lr_M)
        use_toeplitz = True
    else:
        # Batch Random Init
        real_param = (torch.eye(N, device=device).unsqueeze(0) + 0.01 * torch.randn(B, N, N, device=device)).requires_grad_(True)
        imag_param = (0.01 * torch.randn(B, N, N, device=device)).requires_grad_(True)
        optimizer_M = optim.Adam([real_param, imag_param], lr=lr_M)
        use_toeplitz = False

    # 2. Optimization Loop
    print("  -> Running Batched Gradient Descent...")
    
    # We define helper to get normalized M for the whole batch
    def get_batch_M(c_p):
        M_raw = batch_build_M_from_c(c_p, N, device) # (B, N, N)
        # Normalize: M[b, 0, 0] must be 1 real
        # Extract diagonal 0,0: M_raw[:, 0, 0] -> (B,)
        m00 = M_raw[:, 0, 0]
        norm_factor = (m00 / torch.abs(m00)).view(B, 1, 1)
        M_eff = M_raw / norm_factor
        
        # Force real unity (divide by real part of new 0,0)
        m00_new = M_eff[:, 0, 0].real.view(B, 1, 1)
        M_eff = M_eff / m00_new
        return M_eff

    def get_model_cov(M_curr, theta_curr):
        # \M_curr: (B, N, N), theta_curr: (B, P)
        A = steering_vector_batch(N, theta_curr, device=device) # (B, N, P)
        # \R = \M \A \A^H \M^H,  \A \A^H -> (B, N, N)
        AAH = torch.matmul(A, A.conj().transpose(1, 2))
        R = torch.matmul(M_curr, torch.matmul(AAH, M_curr.conj().transpose(1, 2)))
        return R

    def build_M():
        M_raw = batch_build_M_from_c(c_param, N, device) # (B, N, N)
        # Normalize: M[b, 0, 0] must be 1 real
        # Extract diagonal 0,0: M_raw[:, 0, 0] -> (B,)
        m00 = M_raw[:, 0, 0]
        norm_factor = (m00 / torch.abs(m00)).view(B, 1, 1)
        M_eff = M_raw / norm_factor
        
        # Force real unity (divide by real part of new 0,0)
        m00_new = M_eff[:, 0, 0].real.view(B, 1, 1)
        M_eff = M_eff / m00_new
        return M_eff
        
    def compute_loss(R1, R2):
        # Frobenius norm squared per sample: (B, N, N) -> (B,)
        diff = R1 - R2
        return torch.mean(torch.abs(diff)**2, dim=(1, 2)) / Ry_norm

    for outer in range(num_outer):
        
        # Update Theta
        opt_theta = optim.Adam([theta_est], lr=lr_theta)
        for _ in range(num_inner):
            opt_theta.zero_grad()
            A_est = batch_steering_vector(N, theta_est, device) # (B, N, P)
            M_eff = get_batch_M(c_param) # (B, N, N)
            
            # R_model = M @ A @ A^H @ M^H
            # T1 = M @ A -> (B, N, P)
            T1 = torch.matmul(M_eff, A_est)
            # R = T1 @ T1^H -> (B, N, N)
            R_model = torch.matmul(T1, T1.conj().transpose(1, 2))
            
            # Batch Loss
            diff = R_y - R_model
            loss_per_sample = torch.mean(torch.abs(diff)**2, dim=(1, 2)) / Ry_norm
            loss = torch.mean(loss_per_sample) # Mean over batch
            
            loss.backward()
            opt_theta.step()
            with torch.no_grad():
                theta_est.clamp_(-90.0, 90.0)

        # # Update M (c_param)
        # opt_M = optim.Adam([c_param], lr=lr_M)
        # for _ in range(num_inner):
        #     opt_M.zero_grad()
        #     A_est = batch_steering_vector(N, theta_est, device)
        #     M_eff = get_batch_M(c_param)
            
        #     T1 = torch.matmul(M_eff, A_est)
        #     R_model = torch.matmul(T1, T1.conj().transpose(1, 2))
            
        #     diff = R_y - R_model
        #     loss_per_sample = torch.mean(torch.abs(diff)**2, dim=(1, 2)) / Ry_norm
        #     loss = torch.mean(loss_per_sample)
            
        #     loss.backward()
        #     opt_M.step()

        # --- Update M (with Monotone Check) ---
        for _ in range(num_inner):
                
            # Current loss for comparison
            with torch.no_grad():
                M_curr = build_M()
                # cur_losses = compute_loss(R_y, get_model_cov(M_curr, theta_est))
            
            # 2. Step
            optimizer_M.zero_grad()
            M_curr = build_M()
            R_model = get_model_cov(M_curr, theta_est)
            losses = compute_loss(R_y, R_model)
            losses.sum().backward()
            optimizer_M.step()
            
    # Finalize
    theta_est_sorted, _ = torch.sort(theta_est, dim=1)
    M_final = get_batch_M(c_param)
    
    return theta_est_sorted, M_final


# =============================
# 4. Main Execution
# =============================
def run_benchmark():
    dataset_path = os.path.join("DDIM/data/dataset", "test_data_all_snr.pt")
    output_path = os.path.join("DDIM/test_results", "test_results_nonAI_Toe.pt")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset: {dataset_path}")
    full_dataset = torch.load(dataset_path)
    snr_levels = sorted(full_dataset.keys())
    
    final_results = {
        "snr_levels": snr_levels,
        "doa_nmse_avg": [],
        "mcm_nmse_avg": []
    }
    
    print("Starting Batch GPU Benchmark...")
    from em.stable_em_batch import alternating_estimation_monotone_batch

    for snr in snr_levels:
        samples = full_dataset[snr]
        num_samples = len(samples)
        print(f"Processing SNR {snr}dB ({num_samples} samples) as ONE BATCH...")
        
        # 1. Stack Data into Batch Tensor
        # Y_list: List of (N, L) -> Stack to (B, N, L)
        Y_batch = torch.stack([s['Y'] for s in samples]).to(device)
        
        theta_true_batch = torch.stack([s['theta_true'] for s in samples]).to(device)
        M_true_batch = torch.stack([s['M_true'] for s in samples]).to(device)
        
        N = Y_batch.shape[1]
        P = theta_true_batch.shape[1]
        
        # 2. Run Batch Solver
        theta_est_batch, M_est_batch = run_batch_solver(Y_batch, N, P, num_outer=5, num_inner=50, toeplitz_K=5, lr_theta=5e-2, lr_M=1e-2)
        # theta_est_batch, M_est_batch = alternating_estimation_monotone_batch(
        #     Y_batch, N, P,
        #     num_outer=5,
        #     num_inner=50,
        #     lr_theta=0.05,
        #     lr_M=0.01,
        #     toeplitz_K=5,
        #     device=device
        # )
        
        # 3. Compute Metrics (Vectorized)
        # Sort ground truth for comparison
        theta_true_sorted, _ = torch.sort(theta_true_batch, dim=1)
        
        # NMSE Calculation
        # Error norm per sample: (B,)
        err_norm_doa = torch.norm(theta_est_batch - theta_true_sorted, dim=1)
        ref_norm_doa = torch.norm(theta_true_sorted, dim=1)
        nmse_doa_linear = (err_norm_doa / (ref_norm_doa + 1e-8))
        
        err_norm_mcm = torch.norm(M_est_batch - M_true_batch, dim=(1,2))
        ref_norm_mcm = torch.norm(M_true_batch, dim=(1,2))
        nmse_mcm_linear = (err_norm_mcm / (ref_norm_mcm + 1e-8))
        
        # To dB
        nmse_doa_db = 20 * torch.log10(nmse_doa_linear)
        nmse_mcm_db = 20 * torch.log10(nmse_mcm_linear)
        
        # Average over batch
        avg_doa = torch.mean(nmse_doa_db).item()
        avg_mcm = torch.mean(nmse_mcm_db).item()
        
        final_results["doa_nmse_avg"].append(avg_doa)
        final_results["mcm_nmse_avg"].append(avg_mcm)
        
        print(f"  -> Avg DOA NMSE: {avg_doa:.2f} dB")
        print(f"  -> Avg MCM NMSE: {avg_mcm:.2f} dB")
        
    # Save
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
        
    torch.save(final_results, output_path)
    print(f"\nDone! Results saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()