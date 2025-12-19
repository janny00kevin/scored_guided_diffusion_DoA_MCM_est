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