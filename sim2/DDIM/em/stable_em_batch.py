import torch
from em.stable_em import alternating_estimation_monotone

def run_stable_em_on_batch(x0_batch_est, N, P, L, device,
                           num_outer=2, num_inner=50,
                           lr_theta=1e-2, lr_M=1e-2,
                           enforce_M11=True, toeplitz_K=4):
    """
    Performs EM estimation on a batch of denoised signals.
    
    Args:
        x0_batch_est (Tensor): Parallel denoising results with shape (N, S*L).
        N, P, L (int): Physical parameters.
        device: Computing device (e.g., 'cuda' or 'cpu').
        num_outer (int): Number of outer EM iterations.
        num_inner (int): Number of inner optimization steps.
        lr_theta (float): Learning rate for theta.
        lr_M (float): Learning rate for M.
        enforce_M11 (bool): Whether to enforce the M11 constraint.
        toeplitz_K (int): K-parameter for Toeplitz structure.
    
    Returns:
        tuple: (list_theta_est, list_M_est)
            - list_theta_est: List of estimated theta tensors.
            - list_M_est: List of estimated M matrices.
    """
    
    # 1. Infer the number of samples automatically
    total_cols = x0_batch_est.shape[1]
    num_samples = total_cols // L

    # 2. Deparallelize: (N, S*L) -> (N, S, L) -> (S, N, L)
    # This rearranges the batch so we can iterate over individual samples (S).
    x0_est_all = x0_batch_est.reshape(N, num_samples, L).permute(1, 0, 2)

    list_theta_est = []
    list_M_est = []

    # 3. Run EM estimation sample by sample
    for i in range(num_samples):
        x0_sample = x0_est_all[i] # Shape: (N, L)

        theta_est, M_est = alternating_estimation_monotone(
            x0_sample, N, P,
            num_outer=num_outer, 
            num_inner=num_inner,
            lr_theta=lr_theta, 
            lr_M=lr_M,
            enforce_M11=enforce_M11, 
            toeplitz_K=toeplitz_K,
            device=device
        )

        # Move results to CPU to save GPU memory for large batches
        list_theta_est.append(theta_est.detach().cpu())
        list_M_est.append(M_est.detach().cpu())

    return list_theta_est, list_M_est

import torch
import torch.optim as optim
import math

def steering_vector_batch(N, theta_batch, device=None):
    """
    Batch version of steering vector generation.
    Args:
        theta_batch: (Batch_Size, P)
    Returns:
        A: (Batch_Size, N, P)
    """
    # theta_batch: (B, P)
    # k: (B, P)
    theta_rad = theta_batch * (math.pi / 180.0)
    d = 0.5
    k = 2.0 * math.pi * d * torch.sin(theta_rad)
    
    # n: (N,) -> (1, N, 1) to broadcast
    n = torch.arange(0, N, dtype=torch.float32, device=device).view(1, N, 1)
    
    # k: (B, P) -> (B, 1, P)
    k_expanded = k.unsqueeze(1)
    
    # phase: (B, N, P)
    phase = -1j * (n * k_expanded)
    return torch.exp(phase)

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

def alternating_estimation_monotone_batch(x0_batch, N, P,
                                          num_outer=2, num_inner=50,
                                          lr_theta=1e-2, lr_M=1e-2,
                                          enforce_M11=True, toeplitz_K=4, device=None):
    """
    Parallelized version of alternating estimation.
    Args:
        x0_batch: (Batch_Size, N, L)
    Returns:
        theta_est: (Batch_Size, P)
        M_est: (Batch_Size, N, N)
    """
    device = device or x0_batch.device
    B, N_dim, L = x0_batch.shape
    assert N_dim == N
    
    # 1. Compute Sample Covariance: R_y (B, N, N)
    R_y = torch.matmul(x0_batch, x0_batch.conj().transpose(1, 2)) / L
    
    # 2. Initialize Theta using Batch MUSIC (# G angles to search)
    angles = torch.linspace(-90, 90, 181, device=device) # (G,)
    # A_grid: (N, G) -> (1, N, G)
    from data.generator import steering_vector
    A_grid_base = steering_vector(N, angles, device=device) 
    
    # Compute spectrum: P_music = 1 / (\a^H \R^-1 \a)
    R_inv = torch.linalg.pinv(R_y) # (B, N, N)
    
    # Vectorized quadratic form: diag(\A^H \R^-1 \A)
    # Let \H = R\^-1 \A_grid (B, N, N) @ (1, N, G) -> (B, N, G)
    H = torch.matmul(R_inv, A_grid_base.unsqueeze(0))
    # Denominator = sum(conj(\A_grid) * \H, dim=1) -> (B, G)
    denom = torch.sum(A_grid_base.unsqueeze(0).conj() * H, dim=1).real
    spectrum = 1.0 / (denom + 1e-8)
    
    # Top K
    _, idx = torch.topk(spectrum, P, dim=1)
    theta_init = angles[idx] # (B, P)
    
    # Parameters to optimize
    theta_est = theta_init.clone().detach().requires_grad_(True)
    
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
        
    optimizer_theta = optim.Adam([theta_est], lr=lr_theta)
    
    def get_model_cov(M_curr, theta_curr):
        # \M_curr: (B, N, N), theta_curr: (B, P)
        A = steering_vector_batch(N, theta_curr, device=device) # (B, N, P)
        # \R = \M \A \A^H \M^H,  \A \A^H -> (B, N, N)
        AAH = torch.matmul(A, A.conj().transpose(1, 2))
        R = torch.matmul(M_curr, torch.matmul(AAH, M_curr.conj().transpose(1, 2)))
        return R

    def build_M():
        if use_toeplitz:
            return build_M_from_toeplitz_params_batch(c_param, N)
        else:
            return real_param + 1j * imag_param

    def compute_loss(R1, R2):
        # Frobenius norm squared per sample: (B, N, N) -> (B,)
        diff = R1 - R2
        return torch.mean(torch.abs(diff)**2, dim=(1, 2))

    # Alternating Optimization Loop
    for outer in range(num_outer):
        
        # --- Update Theta ---
        for _ in range(num_inner):
            optimizer_theta.zero_grad()
            M_curr = build_M()
            if enforce_M11:
                M_curr = M_curr.clone()
                M_curr[:, 0, 0] = 1.0 + 0j

            R_model = get_model_cov(M_curr, theta_est)
            
            # Sum of losses for batch optimizer
            losses = compute_loss(R_y, R_model)
            total_loss = losses.sum()
            total_loss.backward()
            
            optimizer_theta.step()
            with torch.no_grad():
                theta_est.clamp_(-90.0, 90.0)

        # --- Update M (with Monotone Check) ---
        for _ in range(num_inner):
            # 1. Save current state (data only)
            if use_toeplitz:
                prev_c = c_param.clone().detach()
            else:
                prev_real = real_param.clone().detach()
                prev_imag = imag_param.clone().detach()
                
            # Current loss for comparison
            with torch.no_grad():
                M_curr = build_M()
                cur_losses = compute_loss(R_y, get_model_cov(M_curr, theta_est))
            
            # 2. Step
            optimizer_M.zero_grad()
            M_curr = build_M()
            R_model = get_model_cov(M_curr, theta_est)
            losses = compute_loss(R_y, R_model)
            losses.sum().backward()
            optimizer_M.step()
            
            # 3. Check Monotone Condition
            with torch.no_grad():
                M_new = build_M()
                new_losses = compute_loss(R_y, get_model_cov(M_new, theta_est))
                
                # Identify samples where loss increased
                worse_mask = new_losses > (cur_losses + 1e-12)
                
                if worse_mask.any():
                    # Revert parameters for those samples
                    if use_toeplitz:
                        c_param.data[worse_mask] = prev_c[worse_mask]
                    else:
                        real_param.data[worse_mask] = prev_real[worse_mask]
                        imag_param.data[worse_mask] = prev_imag[worse_mask]

    # Finalize
    with torch.no_grad():
        M_final = build_M()
        if enforce_M11:
            # Force identity on (0,0)
            M_final[:, 0, 0] = 1.0 + 0j
        
        # Sort theta
        theta_final, _ = torch.sort(theta_est, dim=1)
        
    return theta_final.detach(), M_final.detach()