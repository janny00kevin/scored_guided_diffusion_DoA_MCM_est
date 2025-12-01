# This codes generates multiple snapshots of the received signals, create the covariance matrix, but uses a handcrafted
# score-guidance that is derived directly from the loss function || \R_y - \R_{model} ||_F^2, i.e. the guidance equals
# the gradient of this loss wrt the MCM and DOA.
# It estimates the latent variable \x_0 = \C_R \A_R and then by minimizing the Frobenius squared difference between
# the actual covariance matrix and the estimated one (created by the estimated DOAs and MCM), we estimate the DOAs and 
# mutual coupling matrix using gradient descent.  MCM is assumed to be Toeplitz for convenience but the algorithm does not
# assume this parameterization during .

# score_em_euler_maruyama.py
# Score-based (denoising) + Euler-Maruyama guided sampling -> alternating EM for DOA+M
# Requires: torch, numpy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# -----------------------------
# Configuration
# -----------------------------
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Array/problem settings
N = 16       # antennas
P = 3        # sources
L = 500      # snapshots per measurement (columns)
SNR_dB = 10

# Score training settings
num_train = 4000        # simulated training samples (each has L columns)
num_epochs = 30
batch_size = 64        # number of columns per batch
T = 50                 # diffusion timesteps (training)
beta_min = 1e-4
beta_max = 0.02
lr = 1e-3

# Sampling settings
num_steps_sampling = 200   # reverse EM steps (more -> better)
guidance_lambda = 0.8      # guidance strength during sampling
dt = 1.0 / num_steps_sampling

# EM alternating settings (after sampling)
num_outer = 4
num_inner = 200
lr_theta = 5e-2
lr_M = 1e-2

# -----------------------------
# Helpers: complex <-> real stacking
# -----------------------------
def complex_to_real(x):
    # x: (..., N) complex
    return torch.cat([x.real, x.imag], dim=-1)

def real_to_complex(x_real):
    M2 = x_real.shape[-1]
    Nloc = M2 // 2
    return x_real[..., :Nloc] + 1j * x_real[..., Nloc:]

# -----------------------------
# Steering vector & coupling
# -----------------------------
def steering_vector(N, theta_deg, device=None):
    if torch.is_tensor(theta_deg):
        theta = theta_deg.to(dtype=torch.float32)
        dev = theta.device
    else:
        dev = device or torch.device("cpu")
        theta = torch.tensor(theta_deg, dtype=torch.float32, device=dev)
    theta_rad = theta * (np.pi / 180.0)
    d = 0.5
    k = 2.0 * np.pi * d * torch.sin(theta_rad)
    n = torch.arange(0, N, dtype=torch.float32, device=dev)
    phase = -1j * (n[:, None] * k[None, :])
    return torch.exp(phase)

def generate_mutual_coupling_matrix(N, coupling_strength=0.2, c_true=None, device=None):
    dev = device or torch.device("cpu")
    if c_true is None:
        c_true = torch.tensor([1.0, 0.2 + 0.1j, 0.05 + 0.05j], device=dev)
    else:
        c_true = torch.tensor(c_true, dtype=torch.complex64, device=dev)
    M = torch.eye(N, dtype=torch.complex64, device=dev)
    for k in range(1, len(c_true)):
        M += torch.diag(c_true[k] * torch.ones(N - k, dtype=torch.complex64, device=dev), diagonal=k)
        M += torch.diag(c_true[k].conj() * torch.ones(N - k, dtype=torch.complex64, device=dev), diagonal=-k)
    return M

# -----------------------------
# Data generator (simulate many x0 and y)
# -----------------------------
def generate_snapshot_sample(N, P, L, SNR_dB, device):
    theta_true = torch.tensor([-10.0, 20.0, 35.0][:P], dtype=torch.float32, device=device)
    A = steering_vector(N, theta_true, device=device)        # N x P
    M_true = generate_mutual_coupling_matrix(N, 0.2, device=device)
    S = (torch.randn(P, L, dtype=torch.float32, device=device) + 1j * torch.randn(P, L, dtype=torch.float32, device=device)) / math.sqrt(2)
    X = M_true @ (A @ S)     # N x L, complex
    sigma_n = 10 ** (-SNR_dB / 20)
    noise = sigma_n * (torch.randn(N, L, dtype=torch.float32, device=device) + 1j * torch.randn(N, L, dtype=torch.float32, device=device)) / math.sqrt(2)
    Y = X + noise
    return X, Y, theta_true, M_true

# Make training dataset
def make_training_dataset(num_samples):
    X_list = []
    for i in range(num_samples):
        x0, y, _, _ = generate_snapshot_sample(N, P, L, SNR_dB, device)
        X_list.append(x0)  # (N, L) complex
    Xs = torch.stack(X_list, dim=0)  # (S, N, L)
    return Xs

print("Simulating training data...")
Xs = make_training_dataset(num_train)   # (num_train, N, L)
print("Training samples:", Xs.shape)    # num of trianing samples, num. of antennas, num. of snapshots

# -----------------------------
# Precompute diffusion schedules (VP-type)
# -----------------------------
betas = torch.linspace(beta_min, beta_max, T, device=device)         # (T,)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)                           # (T,)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_1m_alpha_bars = torch.sqrt(1.0 - alpha_bars)

# helper q_sample: produce x_t and the noise used
def q_sample(x0_real, t_idx):
    # x0_real: (dim,) real-stacked
    a_bar = sqrt_alpha_bars[t_idx]
    noise = torch.randn_like(x0_real)
    x_t = a_bar * x0_real + sqrt_1m_alpha_bars[t_idx] * noise
    return x_t, noise

# -----------------------------
# Score network (predict score in real domain)
# -----------------------------
class ScoreNet(nn.Module):
    def __init__(self, dim, hidden=512, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, x, t_idx):
        # x: (B, dim), t_idx: (B,) or scalar
        if isinstance(t_idx, int):
            t_in = torch.tensor([[t_idx]], dtype=torch.float32, device=x.device) / float(T)
            te = self.time_emb(t_in).squeeze(0).unsqueeze(0).expand(x.shape[0], -1)
        else:
            t_in = t_idx.float().unsqueeze(-1) / float(T)
            te = self.time_emb(t_in)
        inp = torch.cat([x, te], dim=-1)
        return self.net(inp)

# -----------------------------
# Training loop (denoising score matching)
# -----------------------------
dim = 2 * N
score_net = ScoreNet(dim=dim).to(device)
opt = optim.Adam(score_net.parameters(), lr=lr)

def sample_minibatch_columns(Xs, batch_size):
    S_samples, Nloc, Lloc = Xs.shape
    idx_samples = torch.randint(0, S_samples, (batch_size,), device=device)
    col_idxs = torch.randint(0, Lloc, (batch_size,), device=device)
    x0_batch = torch.stack([Xs[i, :, j] for i, j in zip(idx_samples.tolist(), col_idxs.tolist())], dim=0)  # (B,N)
    x0_real = complex_to_real(x0_batch)
    return x0_real

print("Training score network...")
iters_per_epoch = (num_train * L) // batch_size
for epoch in range(num_epochs):
    total_loss = 0.0
    for it in range(iters_per_epoch):
        x0_real = sample_minibatch_columns(Xs, batch_size)  # (B,2N)
        t_idx = torch.randint(0, T, (batch_size,), device=device)
        x_t = torch.zeros_like(x0_real)
        noise = torch.zeros_like(x0_real)
        for b in range(batch_size):
            x_t[b], noise[b] = q_sample(x0_real[b], int(t_idx[b]))
        # target score = - (x_t - sqrt_alpha_bar * x0) / (1 - alpha_bar)
        # but we have noise: noise = (x_t - sqrt_alpha_bar*x0)/sqrt(1 - alpha_bar)
        # so target_score = - noise / sqrt(1-alpha_bar)
        target_scores = torch.zeros_like(x_t)
        for b in range(batch_size):
            tb = int(t_idx[b])
            target_scores[b] = - (x_t[b] - sqrt_alpha_bars[tb] * x0_real[b]) / (1.0 - alpha_bars[tb])
            # equivalently: - noise[b] / sqrt(1-alpha_bar)
            # target_scores[b] = - noise[b] / (sqrt_1m_alpha_bars[tb])
        pred_scores = score_net(x_t, t_idx)
        loss = torch.mean((pred_scores - target_scores)**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / max(1, iters_per_epoch)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, loss={avg_loss:.6e}")

# -----------------------------
# Euler-Maruyama guided reverse sampler using trained score_net
# -----------------------------
def euler_maruyama_guided_sampler(y_obs_complex, score_net, num_steps=num_steps_sampling,
                                  beta_min=beta_min, beta_max=beta_max,
                                  guidance_lambda=guidance_lambda, dt=dt):
    """
    y_obs_complex: (N, L) complex observed noisy snapshots
    returns x0_est_complex (N, L) complex
    """
    score_net.eval()
    with torch.no_grad():
        Nloc, Lloc = y_obs_complex.shape
        y_cols = [y_obs_complex[:, j] for j in range(Lloc)]
        x0_cols = []
        sigma_y2 = (10 ** (-SNR_dB / 20))**2

        # construct beta schedule for sampling (length = num_steps)
        betas_samp = torch.linspace(beta_min, beta_max, num_steps, device=device)
        alphas_samp = 1.0 - betas_samp
        alpha_bars_samp = torch.cumprod(alphas_samp, dim=0)
        sqrt_alpha_bars_samp = torch.sqrt(alpha_bars_samp)
        sqrt_1m_alpha_bars_samp = torch.sqrt(1.0 - alpha_bars_samp)

        for col in y_cols:
            y_real = complex_to_real(col.unsqueeze(0))[0]  # (2N,)
            # init x_T ~ N(0,I)
            x_t = torch.randn_like(y_real, device=device)

            for i in reversed(range(num_steps)):
                beta = betas_samp[i]
                a_bar = alpha_bars_samp[i]
                sqrt_ab = sqrt_alpha_bars_samp[i]
                sqrt_1m_ab = sqrt_1m_alpha_bars_samp[i]

                # predicted score (real domain)
                t_idx = torch.tensor([i], device=device)
                score_pred = score_net(x_t.unsqueeze(0), t_idx).squeeze(0)  # (2N,)

                # compute denoised x0_hat from score: x0_hat = (x_t + (1 - a_bar) * score) / sqrt_ab
                x0_hat_real = (x_t + (1.0 - a_bar) * score_pred) / (sqrt_ab + 1e-12)

                # grad log-likelihood wrt x0 (assume y = x0 + n)
                grad_x0 = (y_real - x0_hat_real) / (sigma_y2 + 1e-8)

                # map grad_x0 -> grad_xt approximately: dx0/dx_t ≈ 1 / sqrt_ab
                grad_xt = grad_x0 / (sqrt_ab + 1e-12)

                # guided score: add guidance term scaled
                guided = guidance_lambda * grad_xt

                # Euler-Maruyama reverse step
                # drift = -0.5 * beta * x_t - beta * (score + guided)
                drift = -0.5 * beta * x_t - beta * (score_pred + guided)
                if i > 0:
                    z = torch.randn_like(x_t, device=device)
                    x_t = x_t + drift * dt + torch.sqrt(beta * dt) * z
                else:
                    x_t = x_t + drift * dt  # final step without noise

            x0_c = real_to_complex(x_t.unsqueeze(0))[0]
            x0_cols.append(x0_c)

        x0_est = torch.stack(x0_cols, dim=1)  # (N, L) complex
        return x0_est

# -----------------------------
# Alternating EM (covariance-fitting) using denoised x0
# Slight adaptation of your alternating_estimation_from_x0
# -----------------------------
def compute_sample_covariance(Y):
    return (Y @ Y.conj().mT) / Y.shape[1]

def music_init_from_cov(Ry, P, N):
    angles = torch.linspace(-90, 90, 181, device=device)
    A = steering_vector(N, angles, device=device)
    Ry_inv = torch.linalg.pinv(Ry)
    spectrum = torch.zeros(angles.shape[0], device=device)
    for i in range(angles.shape[0]):
        a = A[:, i:i+1]
        spectrum[i] = 1.0 / torch.real((a.conj().mT @ Ry_inv @ a)[0,0])
    _, idx = torch.topk(spectrum, P)
    return angles[idx]

def alternating_estimation_from_x0(x0_init, N, P, num_outer=num_outer, num_inner=num_inner,
                                   lr_theta=lr_theta, lr_M=lr_M):
    Y_like = x0_init
    R_y = compute_sample_covariance(Y_like)
    theta_est = music_init_from_cov(R_y, P, N).clone().detach().requires_grad_(True)
    M_est = generate_mutual_coupling_matrix(N, 0.1, device=device).clone().detach().requires_grad_(True)

    for outer in range(num_outer):
        # update theta
        opt_theta = optim.Adam([theta_est], lr=lr_theta)
        for _ in range(num_inner):
            opt_theta.zero_grad()
            A_est = steering_vector(N, theta_est)
            M_eff = M_est
            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            loss.backward()
            opt_theta.step()
            with torch.no_grad():
                theta_est.clamp_(-90.0, 90.0)
        # update M
        opt_M = optim.Adam([M_est], lr=lr_M)
        for _ in range(num_inner):
            opt_M.zero_grad()
            A_est = steering_vector(N, theta_est)
            R_model = M_est @ A_est @ A_est.conj().mT @ M_est.conj().mT
            loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            loss.backward()
            opt_M.step()

    theta_est, _ = torch.sort(theta_est)
    return theta_est.detach(), M_est.detach()

# -----------------------------
# Run end-to-end: simulate one measurement, sample x0, then EM
# -----------------------------
print("Simulating one measurement...")
X_true, Y_obs, theta_true, M_true = generate_snapshot_sample(N, P, L, SNR_dB, device)

print("Running Euler-Maruyama guided sampling with trained score net...")
x0_est = euler_maruyama_guided_sampler(Y_obs, score_net,
                                       num_steps=num_steps_sampling,
                                       guidance_lambda=guidance_lambda, dt=dt)
print("Sampling done. Running alternating EM on denoised x0...")

theta_est, M_est_out = alternating_estimation_from_x0(x0_est, N, P, num_outer=3, num_inner=200)
print("True DOAs:", theta_true.cpu().numpy())
print("Estimated DOAs:", theta_est.cpu().numpy())

errM = torch.norm(M_est_out - M_true) / torch.norm(M_true)
print("Relative error M:", errM.item())
