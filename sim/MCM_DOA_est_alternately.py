# This codes generates multiple snapshots of the received signals, create the covariance matrix, and directly uses gradient 
# descent to estimate the DOAs (3) and MCM elements (Toeplitz) by minimizing the Frobenius squared difference between
# the actual covariance matrix and the estimated one (created by the estimated DOAs and MCM).  Due to inaccuracy of the
# gradient descent (since the problem is nonconvex), we initialize the initial DOAs using estimates from MUSIC

import torch
import torch.optim as optim
import numpy as np

# =============================
# 1. Configuration
# =============================
torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

N = 16       # number of antennas
P = 3        # number of sources
L = 1000     # snapshots
SNR_dB = 10
torch.manual_seed(0)

# =============================
# 2. Steering vector
# =============================
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

# =============================
# 3. Generate mutual coupling matrix
# =============================
def generate_mutual_coupling_matrix(N, coupling_strength=0.2, c_true=None, device=None):
    dev = device or torch.device("cpu")

    # --------------------------
    # This creates an exponentially decaying value for the MCM elements
    #---------------------------
#    M = torch.eye(N, dtype=torch.complex64, device=dev)
#    for i in range(1, N):
#        for j in range(i, N):
#            val = coupling_strength ** abs(i-j)
#            M[i,j] = val * torch.exp(1j * torch.tensor(np.pi/6))
#            M[j,i] = torch.conj(M[i,j])
#    return M

# -----------------------------
# Toeplitz (banded) model
# -----------------------------
    if c_true is None:
        # Default: first- and second-neighbor coupling
        c_true = torch.tensor([1.0, 0.2 + 0.1j, 0.05 + 0.05j], device=dev)
    else:
        c_true = torch.tensor(c_true, dtype=torch.complex64, device=dev)

    M = torch.eye(N, dtype=torch.complex64, device=dev)
    for k in range(1, len(c_true)):
        M += torch.diag(c_true[k] * torch.ones(N - k, dtype=torch.complex64, device=dev), diagonal=k)
        M += torch.diag(c_true[k].conj() * torch.ones(N - k, dtype=torch.complex64, device=dev), diagonal=-k)
    return M

# =============================
# 4. Generate signals and measurements
# =============================
def generate_measurements(N, P, L, SNR_dB, device):
    true_DOAs = torch.tensor([-10.0, 20.0, 35.0][:P], dtype=torch.float32, device=device)
    A = steering_vector(N, true_DOAs, device=device)
    M_true = generate_mutual_coupling_matrix(N, 0.2, device=device)

    # source signals
    S = (torch.randn(P, L, device=device) + 1j*torch.randn(P, L, device=device)) / np.sqrt(2)
    X = M_true @ (A @ S)

    # noise
    sigma_n = 10 ** (-SNR_dB / 20)
    noise = sigma_n * (torch.randn(N, L, device=device) + 1j*torch.randn(N, L, device=device)) / np.sqrt(2)
    Y = X + noise
    return Y, true_DOAs, M_true



# =============================
# 5. Compute sample covariance
# =============================
def compute_sample_covariance(Y):
    return (Y @ Y.conj().mT) / Y.shape[1]

# =============================
# 6. MUSIC initialization for DOAs
# =============================
def music_initialization(Y, P, N):
    R_y = compute_sample_covariance(Y)
    angles = torch.linspace(-90, 90, 181, device=device)
    A = steering_vector(N, angles, device=device)
    R_y_inv = torch.linalg.pinv(R_y)
    spectrum = torch.zeros(angles.shape[0], device=device)
    for i in range(angles.shape[0]):
        a = A[:, i:i+1]
        spectrum[i] = 1.0 / torch.real((a.conj().mT @ R_y_inv @ a)[0,0])
    _, idx = torch.topk(spectrum, P)
    theta_init = angles[idx]
    return theta_init


# =============================
# 7. Alternating estimation
# =============================
def alternating_estimation(Y, N, P, num_outer=5, num_inner=200, lr_theta=0.05, lr_M=0.01, tol=1e-5):
    prev_loss = float('inf')
    # Initialize DOAs via MUSIC
    theta_est = music_initialization(Y, P, N).clone().detach().requires_grad_(True)
    
    # Initialize M
    M_est = generate_mutual_coupling_matrix(N, 0.1, device=device).clone().detach().requires_grad_(True)
    
    R_y = compute_sample_covariance(Y)

    for outer in range(num_outer):
        # ------------------------
        # Step 1: Fix M, update theta
        # ------------------------
        optimizer_theta = optim.Adam([theta_est], lr=lr_theta)
        for it in range(num_inner):
            optimizer_theta.zero_grad()
            A_est = steering_vector(N, theta_est)
            
            # Reparameterize M to enforce M[0,0] = 1 exactly
            # Step 1: normalize by magnitude and phase
            norm_factor = M_est[0,0] / torch.abs(M_est[0,0])
            M_eff = M_est / norm_factor
            # Step 2: force real unity
            M_eff = M_eff / M_eff[0,0].real

            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            loss.backward()
            optimizer_theta.step()
            
            with torch.no_grad():
                theta_est.clamp_(-90.0, 90.0)
        
        # ------------------------
        # Step 2: Fix theta, update M
        # ------------------------
        optimizer_M = optim.Adam([M_est], lr=lr_M)
        for it in range(num_inner):
            optimizer_M.zero_grad()
            A_est = steering_vector(N, theta_est)
            
            # Reparameterize M to enforce M[0,0] = 1 exactly
            norm_factor = M_est[0,0] / torch.abs(M_est[0,0])
            M_eff = M_est / norm_factor
            M_eff = M_eff / M_eff[0,0].real

            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            loss.backward()
            optimizer_M.step()

        # === loss convergency on ||\R_y - \R_model||_F ===
        with torch.no_grad():
            # current Loss
            A_est = steering_vector(N, theta_est)
            # M normorlized
            norm_factor = M_est[0,0] / torch.abs(M_est[0,0])
            M_eff = M_est / norm_factor
            M_eff = M_eff / M_eff[0,0].real
            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            
            curr_loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            
            # early stopping
            if abs(prev_loss - curr_loss.item()) < tol:
                # print(f"Converged at outer loop {outer}")
                break
            prev_loss = curr_loss.item()

    # Optional: sort DOAs for consistent ordering
    theta_est, _ = torch.sort(theta_est)
    
    # Return the constrained M matrix
    #print('M_eff[0,0] = ', M_eff[0,0])
    norm_factor = M_est[0,0] / torch.abs(M_est[0,0])
    #print('norm factor = ', norm_factor)
    M_eff = M_est / norm_factor
    #print('First M_eff = ', M_eff[0,0])
    M_eff = M_eff / M_eff[0,0].real
    #print('Second M_eff = ', M_eff[0,0]);

    return theta_est.detach(), M_eff.detach()


# =============================
# 8. Run pipeline
# =============================
if __name__ == "__main__":
    Y, true_DOAs, M_true = generate_measurements(N, P, L, SNR_dB, device)
    theta_est, M_est = alternating_estimation(Y, N, P, num_outer=5, num_inner=200)

    print("\nTrue DOAs:", true_DOAs.cpu().numpy())
    print("Estimated DOAs:", theta_est.cpu().numpy())

    #print("\nTrue Mutual Coupling Matrix:\n", M_true)
    #print("\nEstimated Mutual Coupling Matrix:\n", M_est)

    error_norm = torch.norm(M_est - M_true) / torch.norm(M_true)
    print("\nRelative Frobenius norm error of M:", error_norm.item())
