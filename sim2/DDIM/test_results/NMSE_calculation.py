import torch
import os
import scipy.io
import numpy as np

def save_NMSE_as_mat(script_dir, filename, snr_levels, theta_nmse_list, M_nmse_list, x0_nmse_list=None):
    # 0. --- Prepare save path ---
    output_dir = os.path.join(script_dir, 'test_results/NMSE_raw_mats')
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
        'x0_nmse': x0_arr if x0_nmse_list is not None else []
    })

    print(f"[Info] NMSE results saved to test_results/NMSE_raw_mats/{filename}")
class NMSEAccumulator:
    def __init__(self, power_offset_db=0.0):
        """
        Accumulates error metrics across batches to compute global averages.
        """
        self.power_offset_db = power_offset_db
        
        # Accumulators
        self.x0_nmse_sum = 0.0
        self.theta_nmse_sum = 0.0
        self.M_nmse_sum = 0.0
        self.total_samples = 0

    def update(self, x0_est, x0_true, theta_est, theta_true, M_est, M_true):
        """
        Process a batch of estimates and ground truths.
        All inputs should be Tensors.
        """
        batch_size = x0_est.shape[0]
        self.total_samples += batch_size

        # --- 1. x0 Metric (Linear Sum) ---
        # NMSE = ||x - x_hat||^2 / ||x||^2
        x0_nmse = torch.norm(x0_est - x0_true, dim=(1, 2))**2 / torch.norm(x0_true, dim=(1, 2))**2
        # Accumulate SUM of linear errors (will divide by N later)
        self.x0_nmse_sum += torch.sum(x0_nmse).item()

        # --- 2. Theta Metric ---
        # Sort angles first
        theta_true_sorted, _ = torch.sort(theta_true, dim=1)
        theta_est_sorted, _ = torch.sort(theta_est, dim=1)
        
        theta_err = torch.norm(theta_true_sorted - theta_est_sorted, p=2, dim=1)**2
        theta_ref = torch.norm(theta_true_sorted, p=2, dim=1)**2
        theta_nmse = theta_err / (theta_ref + 1e-8)
        # Accumulate SUM of dB values 
        self.theta_nmse_sum += (torch.sum(theta_nmse + 1e-8)).item()

        # --- 3. M Matrix Metric (Log Sum) ---
        M_err = torch.norm(M_true - M_est, p='fro', dim=(1, 2))**2
        M_ref = torch.norm(M_true, p='fro', dim=(1, 2))**2
        M_nmse = M_err / (M_ref + 1e-8)
        # Accumulate SUM of dB values
        self.M_nmse_sum += torch.sum(M_nmse + 1e-8).item()

    def get_final_metrics(self):
        """
        Returns the final averaged metrics in dB.
        returns: (x0_db, theta_db, M_db)
        """

        # x0: Arithmetic Mean of Linear NMSE -> converted to dB
        avg_x0_linear = self.x0_nmse_sum / self.total_samples
        x0_db = 10 * np.log10(avg_x0_linear) + self.power_offset_db

        # Theta & M: Arithmetic Mean of Linear NMSE 
        theta_db = 10 * np.log10(self.theta_nmse_sum / self.total_samples)
        M_db = 10 * np.log10(self.M_nmse_sum / self.total_samples)

        print(f"  [X0]    NMSE: {x0_db:.2f} dB")
        print(f"  [Theta] NMSE: {theta_db:.2f} dB")
        print(f"  [M Mat] NMSE: {M_db:.2f} dB")

        return x0_db, theta_db, M_db