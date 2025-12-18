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
from data.data_loader import get_or_create_dataset
from diffusion.ddim_sampler_parallel import ddim_epsnet_guided_sampler_batch
from em.stable_em import alternating_estimation_monotone

# import train function from train.py
from train import train_epsilon_net

# -----------------------------
# Configurations
# -----------------------------
N=16         # N: # of antennas
P=3          # P: # of paths/sources
L=128        # L: # of snapshots (how many we collect \y)
SNR_dB=10

# Training settings
CUDA = 1
NUM_EPOCHS = 50
BATCH_SIZE = 4096
LR = 1e-4
MODEL_TYPE = 'mlp'
NUM_TRAIN_SAMPLES = int(5000)  # try 1e5

# Difussion process settings
BETA_MIN=1e-4
BETA_MAX=0.02
T_DIFFUSION=1000.0

device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Load/generate training data
# -----------------------------
Xs_train = get_or_create_dataset(NUM_TRAIN_SAMPLES, N, P, L, device, script_dir, use_toeplitz=True)

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
# Test on a single measurement
# -----------------------------
print('Simulating one test measurement...')
X_true, Y_obs, theta_true, M_true, _ = generate_snapshot_sample(N, P, L, SNR_dB, device,
                                                                 randomize=False, use_toeplitz=True)

print('Running batch DDIM guided sampler (trained net)...')
x0_est = ddim_epsnet_guided_sampler_batch(Y_obs.to(device), eps_net, num_steps=50, T=50.0, guidance_lambda=0.8,
                                          device=device, apply_physics_projection=True)

print('Running stable alternating EM on denoised x0...')
theta_est, M_est = alternating_estimation_monotone(x0_est, N, P,
                                                   num_outer=2, num_inner=50,
                                                   lr_theta=1e-2, lr_M=1e-2,
                                                   enforce_M11=True, toeplitz_K=4,
                                                   device=device)

# -----------------------------
# Report results
# -----------------------------
print('True DOAs:', theta_true.cpu().numpy())
print('Estimated DOAs:', theta_est.cpu().numpy())
print('Relative error DOA:', (torch.norm(theta_est - theta_true) / torch.norm(theta_true)).item())
print('\n')
print('Ground truth M = ', M_true)
print('M_est = ', M_est)
print('Relative error M:', (torch.norm(M_est - M_true) / torch.norm(M_true)).item())