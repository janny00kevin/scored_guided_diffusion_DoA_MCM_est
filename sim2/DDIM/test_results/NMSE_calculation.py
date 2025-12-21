import torch

def calculate_nmse_theta_M(theta_est, M_est, theta_true, M_true, snr, device=None):
    if device is None:
        device = theta_est.device
    # theta_true = full_dataset['theta_true'].to(device) # (num_samples, P)
    # M_true = full_dataset['M_true'].to(device)       # (num_samples, N, N)

    # Sort the true and estimated theta for the comparison
    theta_true_sorted, _ = torch.sort(theta_true, dim=1)
    theta_est_sorted, _ = torch.sort(theta_est, dim=1)

    # 1. --- Theta NMSE ---
    theta_error = torch.norm(theta_true_sorted - theta_est_sorted, p=2, dim=1)
    theta_ref = torch.norm(theta_true_sorted, p=2, dim=1)
    nmse_per_sample = theta_error / (theta_ref + 1e-8)
    theta_nmse_db = torch.mean(20 * torch.log10(nmse_per_sample + 1e-8))

    # 2. --- M Matrix NMSE ---
    M_error = torch.norm(M_true - M_est, p='fro', dim=(1, 2))**2
    M_ref = torch.norm(M_true, p='fro', dim=(1, 2))**2
    M_nmse_per_sample = M_error / (M_ref + 1e-8)
    M_nmse_db = torch.mean(10 * torch.log10(M_nmse_per_sample + 1e-8))

    # 3. --- Print Results ---
    num_samples = theta_est.shape[0]
    print(f"Results for SNR {snr} dB (Avg over {num_samples} samples):")
    print(f"  [Theta] NMSE: {theta_nmse_db.item():.2f} dB")
    print(f"  [M Mat] NMSE: {M_nmse_db.item():.2f} dB")
    
    return theta_nmse_db.item(), M_nmse_db.item()