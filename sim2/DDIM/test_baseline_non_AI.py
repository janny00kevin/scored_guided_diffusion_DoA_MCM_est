import torch
import os
from data.data_loader import get_or_create_testing_dataset
from em.stable_em_batch import alternating_estimation_monotone_batch
from test_results.NMSE_calculation import calculate_nmse_theta_M, save_NMSE_as_mat, calculate_nmse_x0

# -----------------------------
# Configurations
# -----------------------------
N = 16         # N: # of antennas
P = 3          # P: # of paths/sources
L = 128        # L: # of snapshots
SNR_LEVELS = [-4, -2, 0, 2, 4, 6, 8, 10]
NUM_TEST_SAMPLES = 3000

# EM parameters
NUM_OUTER_EM = 10      
NUM_INNER_EM = 5     
LR_THETA = 5e-2
LR_M = 1e-2
TOEPLITZ_K = 5        

RESULT_FILE_NAME = "NMSE_Baseline_non_AI.mat"

# -----------------------------
# Setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)

def run_baseline_test():
    # 1. --- Load/generate testing data ---
    full_dataset = get_or_create_testing_dataset(NUM_TEST_SAMPLES, N, P, L, SNR_LEVELS,
                                                 device, script_dir, use_toeplitz=True)

    print(f'[Info] Starting Baseline Test (No DDIM Denoising)...')

    theta_nmse_results = []
    M_nmse_results = []
    x0_nmse_results = []

    for snr in SNR_LEVELS:
        print(f"\n--- Processing SNR = {snr} dB ---")

        # 2. --- Load Ys for this SNR level, shape: (Num_Samples, N, L) ---
        Ys_obs = full_dataset['observations'][snr].to(device)
        # calculate NMSE of x0 directly from observations (-SNR)
        x0_nmse = calculate_nmse_x0(Ys_obs, full_dataset['X_clean'].to(device), device=device)
        x0_nmse_results.append(x0_nmse)
        
        # 3. --- Run EM estimation on the observed data directly ---
        theta_est_batch, M_est_batch = alternating_estimation_monotone_batch(
                                            Ys_obs, N, P,
                                            num_outer=NUM_OUTER_EM, 
                                            num_inner=NUM_INNER_EM,
                                            lr_theta=LR_THETA, 
                                            lr_M=LR_M,
                                            toeplitz_K=5,
                                            device=device)

        # 4. --- Calculate NMSE for each SNR level ---
        theta_nmse_db, M_nmse_db = calculate_nmse_theta_M(theta_est_batch, M_est_batch,
                                                            full_dataset['theta_true'].to(device),
                                                            full_dataset['M_true'].to(device),
                                                            snr, device=device)
        theta_nmse_results.append(theta_nmse_db)
        M_nmse_results.append(M_nmse_db)

    # 5. --- Save results to .mat file ---
    save_NMSE_as_mat(script_dir, RESULT_FILE_NAME, SNR_LEVELS, theta_nmse_results, M_nmse_results, x0_nmse_results)
    print(f"\n[Info] Done. Results saved to test_results/{RESULT_FILE_NAME}")

if __name__ == "__main__":
    run_baseline_test()