import torch
import os
import scipy.io
import numpy as np

def calculate_nmse_x0(x0_est, x0_true, device=None):
    if device is None:
        device = x0_est.device
    x0_est = x0_est.to(device)
    x0_true = x0_true.to(device)

    # To compensate for the power scaling in data generation
    POWER_OFFSET_DB = 10 * np.log10(3.0) 

    # calculate NMSE, Shape: (Batch, N, L)
    diff = x0_est - x0_true
    norm_diff = torch.norm(diff, dim=(1, 2)) ** 2
    norm_true = torch.norm(x0_true, dim=(1, 2)) ** 2
    nmse_linear = norm_diff / (norm_true + 1e-12)
    
    # turn to dB
    nmse_db = 10 * np.log10(torch.mean(nmse_linear).item()) + POWER_OFFSET_DB
    
    print(f"  [X0]    NMSE: {nmse_db:.2f} dB")
    return nmse_db

def calculate_nmse_theta_M(theta_est, M_est, theta_true, M_true, snr, device=None):
    if device is None:
        device = theta_est.device

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
    # print(f"Results for SNR {snr} dB (Avg over {num_samples} samples):")
    print(f"  [Theta] NMSE: {theta_nmse_db.item():.2f} dB")
    print(f"  [M Mat] NMSE: {M_nmse_db.item():.2f} dB")
    
    return theta_nmse_db.item(), M_nmse_db.item()

def save_NMSE_as_mat(script_dir, filename, snr_levels, theta_nmse_list, M_nmse_list, x0_nmse_list=None):
    # 0. --- Prepare save path ---
    output_dir = os.path.join(script_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    # 1. --- Turn Torch tensor into Numpy Array ---
    snr_arr = np.array(snr_levels)
    theta_arr = np.array(theta_nmse_list)
    M_arr = np.array(M_nmse_list)
    if x0_nmse_list is not None:
        x0_arr = np.array(x0_nmse_list)

    # 2. --- Save .mat file (for future use) ---
    scipy.io.savemat(save_path, {
        'snr_range': snr_arr,
        'theta_nmse': theta_arr,
        'M_nmse': M_arr,
        # 'x0_nmse': x0_arr if x0_nmse_list is not None else []
    })

    print(f"[Info] NMSE results saved to test_results/{filename}")
