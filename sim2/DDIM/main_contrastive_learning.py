import torch
import os
# -----------------------------
# Configurations
# -----------------------------
RUN_ID = 2
MODE = {1: 'train', 2: 'test'}.get(RUN_ID, 'train')

N=16         # N: # of antennas
P=3          # P: # of paths/sources
L=128        # L: # of snapshots (how many we collect \y)
SNR_LEVELS=[-4, -2, 0, 2, 4, 6, 8, 10]
CUDA = 1

# Training settings
NUM_EPOCHS = 10000
TRAIN_BATCH_SIZE = 1024
LR = 1e-3
MODEL_TYPE = 'mlp'  # 'unet1d' or 'mlp'
NUM_TRAIN_SAMPLES = int(5000)  # try 1e5
NUM_TEST_SAMPLES = int(3000)    
VAL_SPLIT = 0.1
PATIENCE = 15
contrastive_weight = 0.1
temperature = 0.2

# Difussion process settings
BETA_MIN=1e-4
BETA_MAX=0.02
T_DIFFUSION=1000.0
NUM_SAMPLING_STEPS=50
GUIDANCE_LAMBDA=1.1

# testing settings
TEST_BATCH_SIZE = 3000
MODEL_WEIGHT_FILE_NAME = f"CL{contrastive_weight:.0e}_same_x0_temp{temperature:.0e}_{MODEL_TYPE}_lr{LR:.0e}.pth"
NMSE_RESULT_FILE_NAME = f"NMSE_{MODEL_WEIGHT_FILE_NAME.split('.')[0]}.mat"

# -----------------------------

device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)

# -----------------------------
# Training part
# -----------------------------
if MODE == 'train':
    from data.data_loader import get_or_create_training_dataset
    from train import train_latent_epsnet
    # -----------------------------
    # Load/generate training data
    # -----------------------------
    Xs_train = get_or_create_training_dataset(NUM_TRAIN_SAMPLES, N, P, L, device, script_dir, use_toeplitz=True)

    # -----------------------------
    # Train diffusion eps-net
    # -----------------------------
    print('[Info] Training epsilon net...')
    # Original: 'unet1d'
    eps_net = train_latent_epsnet(Xs_train, MODEL_TYPE, NUM_EPOCHS, TRAIN_BATCH_SIZE, LR,
                                BETA_MIN, BETA_MAX, T_DIFFUSION, 
                                VAL_SPLIT, PATIENCE,
                                device, script_dir, MODEL_WEIGHT_FILE_NAME,
                                contrastive_weight, temperature)

# -----------------------------
# Testing part
# -----------------------------
elif MODE == 'test':
    from data.data_loader import get_or_create_testing_dataset
    from models.eps_net_loader import load_trained_model
    from diffusion.ddim_sampler_parallel import ddim_epsnet_guided_sampler_dynamic
    from em.stable_em_batch import alternating_estimation_monotone_batch
    from test_results.NMSE_calculation import NMSEAccumulator, save_NMSE_as_mat
    import numpy as np

    # --- Load/generate testing data ---
    full_dataset = get_or_create_testing_dataset(NUM_TEST_SAMPLES, N, P, L, SNR_LEVELS,
                                                device, script_dir, use_toeplitz=True)

    print(f'[Info] Loading model...')
    eps_net, data_mean, data_std = load_trained_model(script_dir, device, N, MODEL_TYPE, MODEL_WEIGHT_FILE_NAME, use_CL=True)

    theta_nmse_results = []
    M_nmse_results = []
    x0_nmse_results = []
    POWER_OFFSET_DB = 10 * np.log10(3.0)

    for snr in SNR_LEVELS:
        print(f"\n--- Processing SNR = {snr} dB for {NUM_TEST_SAMPLES} samples ---")
        # Initialize NMSE Accumulators for this SNR
        metric_tracker = NMSEAccumulator(power_offset_db=POWER_OFFSET_DB)

        # Load all data for this SNR
        Ys_all = full_dataset['observations'][snr]
        X_clean_all = full_dataset['X_clean']
        theta_true_all = full_dataset['theta_true']
        M_true_all = full_dataset['M_true']

        num_total_samples = Ys_all.shape[0]

        # --- Mini-Batch Loop ---
        for i in range(0, num_total_samples, TEST_BATCH_SIZE):
            # 1. Prepare Batch, Reshape for Parallel DDIM Sampling: (B, N, L) -> (N, B * L)
            indices = slice(i, min(i + TEST_BATCH_SIZE, num_total_samples))
            Ys_batch = Ys_all[indices].to(device)
            Ys_flat = Ys_batch.permute(1, 0, 2).reshape(N, -1)

            # --- 1. denoising using DDIM guided sampler (N, B * L) -> (N, B * L) ---
            with torch.no_grad():
                x0_flat = ddim_epsnet_guided_sampler_dynamic(
                    Ys_flat, eps_net, MODEL_TYPE, snr,
                    data_mean, data_std,
                    NUM_SAMPLING_STEPS, T_DIFFUSION, BETA_MIN, BETA_MAX, GUIDANCE_LAMBDA,
                    device=device, apply_physics_projection=True
                )
            x0_est = x0_flat.reshape(N, -1, L).permute(1, 0, 2) # (B, N, L)

        # # Calculate NMSE of \x0_hat
        # x0_nmse = calculate_nmse_x0(x0_est_all, full_dataset['X_clean'].to(device),device=device)

            # --- 2. Estimate theta and \C_R using EM algorithm ---
            theta_est, M_est = alternating_estimation_monotone_batch(
                                    x0_est, N, P,
                                    num_outer=10, num_inner=5,
                                    lr_theta=5e-2, lr_M=1e-2,
                                    toeplitz_K=5, device=device
                                )
            # Update NMSE metrics
            metric_tracker.update(
                x0_est, X_clean_all[indices].to(device),
                theta_est, theta_true_all[indices].to(device),
                M_est, M_true_all[indices].to(device)
            )

        # Calculate NMSE for each SNR level
        x0_db, theta_db, M_db = metric_tracker.get_final_metrics()
        
        x0_nmse_results.append(x0_db)
        theta_nmse_results.append(theta_db)
        M_nmse_results.append(M_db)

    save_NMSE_as_mat(script_dir, NMSE_RESULT_FILE_NAME, SNR_LEVELS, theta_nmse_results, M_nmse_results, x0_nmse_results)
    