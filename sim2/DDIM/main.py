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

# Training settings
CUDA = 1
NUM_EPOCHS = 50
BATCH_SIZE = 4096
LR = 1e-4
MODEL_TYPE = 'mlp'
NUM_TRAIN_SAMPLES = int(5000)  # try 1e5
NUM_TEST_SAMPLES = int(30)    

# Difussion process settings
BETA_MIN=1e-4
BETA_MAX=0.02
T_DIFFUSION=1000.0
NUM_SAMPLING_STEPS=50
GUIDANCE_LAMBDA=0.4

# testing settings
MODEL_WEIGHT_FILE_NAME = "DDIM_ep50_lr1e-04_t1000_bmax2e-02.pth"

# -----------------------------

device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))

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
    from diffusion.ddim_sampler_parallel import ddim_epsnet_guided_sampler_batch
    from em.stable_em import alternating_estimation_monotone
    from models.eps_net_loader import load_trained_model
    from em.stable_em_batch import run_stable_em_on_batch
    from em.stable_em_batch import alternating_estimation_monotone_batch

    # -----------------------------
    # Load/generate testing data
    # -----------------------------

    full_dataset = get_or_create_testing_dataset(NUM_TEST_SAMPLES, N, P, L, SNR_LEVELS,
                                                device, script_dir, use_toeplitz=True)

    print(f'[Info] Loading model...')
    eps_net = load_trained_model(script_dir, device, N, MODEL_TYPE, MODEL_WEIGHT_FILE_NAME)

    # results = {
    #     "doa_est": [],
    #     "mcm_est": [],
    #     "doa_true": full_dataset['theta_true'],
    #     "mcm_true": full_dataset['M_true'],
    #     "snr_levels": SNR_LEVELS
    # }


    for snr in SNR_LEVELS:
        print(f"\n--- Processing SNR = {snr} dB ---")

        # Load Ys for this SNR level, shape: (Num_Samples, N, L)
        Ys_obs = full_dataset['observations'][snr].to(device)
        num_samples = Ys_obs.shape[0]

        # =================================================================
        # Reshape for Parallel DDIM Sampling: (S, N, L) -> (N, S * L)
        # Thus the Sampler will treat it as N antennas, but with S*L snapshots of a very large matrix
        # =================================================================

        # Parallelize: permute: (S, N, L) -> (N, S, L)  reshape: (N, S, L) -> (N, S * L)
        Ys_batch = Ys_obs.permute(1, 0, 2).reshape(N, -1)

        # denoising using DDIM guided sampler (N, S * L)
        x0_batch_est = ddim_epsnet_guided_sampler_batch(Ys_batch, eps_net, snr,
                                NUM_SAMPLING_STEPS, T_DIFFUSION, BETA_MIN, BETA_MAX, GUIDANCE_LAMBDA,
                                device=device, apply_physics_projection=False)

        list_theta_est, list_M_est = run_stable_em_on_batch(x0_batch_est, N, P, L, device,
                                            num_outer=2, num_inner=50,
                                            lr_theta=0.05, lr_M=1e-2,
                                            enforce_M11=True, toeplitz_K=4)
        

        # theta_est_batch, M_est_batch = alternating_estimation_monotone_batch(
        #                                     x0_batch_est, N, P,
        #                                     num_outer=5, 
        #                                     num_inner=50,
        #                                     lr_theta=1e-2, 
        #                                     lr_M=1e-2,
        #                                     toeplitz_K=4,
        #                                     device=device
        #                                 )
        
        theta_true = full_dataset['theta_true'].to(device) # (num_samples, P)
        M_true = full_dataset['M_true'].to(device)       # (num_samples, N, N)
        # 確保真實值與估計值都已排序（避免對應錯誤）
        theta_true_sorted, _ = torch.sort(theta_true, dim=1)

        theta_est_tensor = torch.stack(list_theta_est).to(device)
        M_est_tensor = torch.stack(list_M_est).to(device)

        theta_error = torch.norm(theta_true_sorted - theta_est_tensor, p=2, dim=1)**2
        theta_ref = torch.norm(theta_true_sorted, p=2, dim=1)**2
        theta_nmse_linear = torch.mean(theta_error / theta_ref)
        theta_nmse_db = 10 * torch.log10(theta_nmse_linear)

        # --- M Matrix NMSE ---
        # 使用 Frobenius Norm 計算矩陣誤差
        M_error = torch.norm(M_true - M_est_tensor, p='fro', dim=(1, 2))**2
        M_ref = torch.norm(M_true, p='fro', dim=(1, 2))**2
        M_nmse_linear = torch.mean(M_error / M_ref)
        M_nmse_db = 10 * torch.log10(M_nmse_linear)

        # 4. 打印結果
        print(f"Results for SNR {snr} dB (Avg over {num_samples} samples):")
        print(f"  [Theta] NMSE: {theta_nmse_db.item():.2f} dB")
        print(f"  [M Mat] NMSE: {M_nmse_db.item():.2f} dB")
        # print()


    # -----------------------------
    # Test on a single measurement
    # -----------------------------
    print('Simulating one test measurement...')
    # X_true, Y_obs, theta_true, M_true, _ = generate_snapshot_sample(N, P, L, SNR_dB, device,
    #                                                                  randomize=False, use_toeplitz=True)

    # print('Running batch DDIM guided sampler (trained net)...')
    # x0_est = ddim_epsnet_guided_sampler_batch(Y_obs.to(device), eps_net, num_steps=50, T=50.0, guidance_lambda=0.8,
    #                                         device=device, apply_physics_projection=True)

    # print('Running stable alternating EM on denoised x0...')
    # theta_est, M_est = alternating_estimation_monotone(x0_est, N, P,
    #                                                 num_outer=2, num_inner=50,
    #                                                 lr_theta=1e-2, lr_M=1e-2,
    #                                                 enforce_M11=True, toeplitz_K=4,
    #                                                 device=device)

    # -----------------------------
    # Report results
    # -----------------------------
    # print('True DOAs:', theta_true.cpu().numpy())
    # print('Estimated DOAs:', theta_est.cpu().numpy())
    # print('Relative error DOA:', (torch.norm(theta_est - theta_true) / torch.norm(theta_true)).item())
    # print('\n')
    # print('Ground truth M = ', M_true)
    # print('M_est = ', M_est)
    # print('Relative error M:', (torch.norm(M_est - M_true) / torch.norm(M_true)).item())