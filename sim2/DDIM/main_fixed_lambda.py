# This code generates multiple snapshots of the received signals, create the covariance matrix, and train
# the model using these multiple snapshots at once.  The model can be a UNet or MLP (default is UNet) 
# that is used to find the latent variable \x_0 = \C_R \A_R \s + \n, where \C_R (denoted as M in the 
# code) is the angle independent MCM.  The array factor matrix \A_R only contains the azimuth angles
# (3 angles) listed from smalllest to highest.  This is because the DOAs are estimated blindly, so
# there is no good way for the estimates to corresponding directly with the angles to measure square
# array unless I sort the ground-truth and estimated angle in ascending order.  
#
# After \x0_hat is obtained, this is treated as the "complete data" in the EM algorithm, which is then
# used to estimate the DOAs and \C_R by minimizing || \R_{\x0_hat} - \R_{model} ||_F^2 using alterating
# minimization.  \R_{model} is the covariance matrix concerning the DOA and \C_R estimates.  Note that
# the gradient descent used to find the DOAs are initiaized using MUSIC because with random 
# initialization, the DOA estimates are very bad.  The \C_R is also obtained using gradient descent
# and the estimate assumes knowledge about \C_R being a Toeplitz matrix.


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
NUM_EPOCHS = 50
BATCH_SIZE = 4096
LR = 1e-4
MODEL_TYPE = 'mlp'
NUM_TRAIN_SAMPLES = int(5000)  # try 1e5
NUM_TEST_SAMPLES = int(3000)    

# Difussion process settings
BETA_MIN=1e-4
BETA_MAX=0.02
T_DIFFUSION=1000.0
NUM_SAMPLING_STEPS=50
GUIDANCE_LAMBDA=0.25

# testing settings
MODEL_WEIGHT_FILE_NAME = f"DDIM_ep{NUM_EPOCHS}_lr{LR:.0e}_t{int(T_DIFFUSION)}_bmax{BETA_MAX:.0e}.pth"
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
    from train import train_epsilon_net
    # -----------------------------
    # Load/generate training data
    # -----------------------------
    Xs_train = get_or_create_training_dataset(NUM_TRAIN_SAMPLES, N, P, L, device, script_dir, use_toeplitz=True)

    # -----------------------------
    # Train diffusion eps-net
    # -----------------------------
    print('[Info] Training epsilon net...')
    # Original: 'unet1d'
    eps_net = train_epsilon_net(Xs_train, MODEL_TYPE, 
                                NUM_EPOCHS, BATCH_SIZE, LR,
                                BETA_MIN, BETA_MAX, T_DIFFUSION, 
                                device, script_dir)

# -----------------------------
# Testing part
# -----------------------------
elif MODE == 'test':
    from data.data_loader import get_or_create_testing_dataset
    from models.eps_net_loader import load_trained_model
    from diffusion.ddim_sampler_parallel import ddim_epsnet_guided_sampler_batch
    from em.stable_em_batch import alternating_estimation_monotone_batch
    from test_results.NMSE_calculation import calculate_nmse_x0, calculate_nmse_theta_M, save_NMSE_as_mat

    # -----------------------------
    # Load/generate testing data
    # -----------------------------

    full_dataset = get_or_create_testing_dataset(NUM_TEST_SAMPLES, N, P, L, SNR_LEVELS,
                                                device, script_dir, use_toeplitz=True)

    print(f'[Info] Loading model...')
    eps_net, data_mean, data_std = load_trained_model(script_dir, device, N, MODEL_TYPE, MODEL_WEIGHT_FILE_NAME)

    theta_nmse_results = []
    M_nmse_results = []
    x0_nmse_results = []

    for snr in SNR_LEVELS:
        # if abs(snr - 10) > 6.1: 
        #     continue
        print(f"\n--- Processing SNR = {snr} dB for {NUM_TEST_SAMPLES} samples ---")

        # Load Ys for this SNR level, shape: (Num_Samples, N, L)
        Ys_obs = full_dataset['observations'][snr].to(device)
        num_samples = Ys_obs.shape[0]

        # =================================================================
        # Reshape for Parallel DDIM Sampling: (S, N, L) -> (N, S * L)
        # Thus the Sampler will treat it as N antennas, but with S*L snapshots of a large matrix
        # =================================================================

        # Parallelize: permute: (S, N, L) -> (N, S, L)  reshape: (N, S, L) -> (N, S * L)
        Ys_batch = Ys_obs.permute(1, 0, 2).reshape(N, -1)

        # --- 1. denoising using DDIM guided sampler (N, S * L) -> (N, S * L) ---
        x0_batch_est = ddim_epsnet_guided_sampler_batch(Ys_batch, eps_net, snr,
                                data_mean, data_std,
                                NUM_SAMPLING_STEPS, T_DIFFUSION, BETA_MIN, BETA_MAX, GUIDANCE_LAMBDA,
                                device=device, apply_physics_projection=True)

        # Deparallelize: x0_batch_est: (N, S * L) -> (N, S, L)-> (S, N, L) : x0_est_all
        x0_est_all = x0_batch_est.reshape(N, num_samples, L).permute(1, 0, 2)

        # Calculate NMSE of \x0_hat
        x0_nmse = calculate_nmse_x0(x0_est_all, full_dataset['X_clean'].to(device),device=device)

        # --- 2. Estimate theta and \C_R using EM algorithm ---
        theta_est_batch, M_est_batch = alternating_estimation_monotone_batch(
                                            x0_est_all, N, P,
                                            num_outer=5, num_inner=50,
                                            lr_theta=5e-2, lr_M=1e-2,
                                            toeplitz_K=5, device=device)

        # Calculate NMSE for each SNR level
        theta_nmse_db, M_nmse_db = calculate_nmse_theta_M(theta_est_batch, M_est_batch,
                                                            full_dataset['theta_true'].to(device),
                                                            full_dataset['M_true'].to(device),
                                                            snr, device=device)
        
        x0_nmse_results.append(x0_nmse)
        theta_nmse_results.append(theta_nmse_db)
        M_nmse_results.append(M_nmse_db)

    save_NMSE_as_mat(script_dir, NMSE_RESULT_FILE_NAME, SNR_LEVELS, theta_nmse_results, M_nmse_results, x0_nmse_results)
    