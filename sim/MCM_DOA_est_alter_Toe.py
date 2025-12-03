import torch
import torch.optim as optim
import numpy as np

# =============================
# 1. Configuration
# =============================
torch.set_default_dtype(torch.float32)
# Note: Defaults to cuda, but will fallback to cpu if specified externally or unavailable.
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
    if c_true is None:
        c_true = torch.tensor([1.0, 0.2 + 0.1j, 0.05 + 0.05j], device=dev)
    else:
        c_true = torch.tensor(c_true, dtype=torch.complex64, device=dev)

    M = torch.eye(N, dtype=torch.complex64, device=dev)
    for k in range(1, len(c_true)):
        M += torch.diag(c_true[k] * torch.ones(N - k, dtype=torch.complex64, device=dev), diagonal=k)
        M += torch.diag(c_true[k].conj() * torch.ones(N - k, dtype=torch.complex64, device=dev), diagonal=-k)
    return M

# =============================
# [New] Helper: Build Toeplitz M from vector
# =============================
def build_M_from_toeplitz_params(c_param, N, device):
    """
    c_param: Complex vector containing the first column of the Toeplitz matrix.
             Shape (K,), where K is the bandwidth (usually small, e.g., 4 or 5).
    """
    K = c_param.shape[0]
    
    # c_param[0] is the main diagonal.
    # We construct the matrix starting with the main diagonal.
    M = torch.diag(c_param[0] * torch.ones(N, dtype=torch.complex64, device=device))
    
    # Loop to add upper and lower diagonals based on the bandwidth K
    for k in range(1, min(K, N)):
        val = c_param[k]
        # Upper diagonal
        M += torch.diag(val * torch.ones(N - k, dtype=torch.complex64, device=device), diagonal=k)
        # Lower diagonal (Hermitian symmetry)
        M += torch.diag(val.conj() * torch.ones(N - k, dtype=torch.complex64, device=device), diagonal=-k)
        
    return M

# =============================
# 4. Generate signals and measurements
# =============================
def generate_measurements(N, P, L, SNR_dB, device):
    true_DOAs = torch.tensor([-10.0, 20.0, 35.0][:P], dtype=torch.float32, device=device)
    A = steering_vector(N, true_DOAs, device=device)
    M_true = generate_mutual_coupling_matrix(N, 0.2, device=device)
    
    S = (torch.randn(P, L, device=device) + 1j*torch.randn(P, L, device=device)) / np.sqrt(2)
    X = M_true @ (A @ S)
    
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
# 6. MUSIC initialization
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
# 7. Alternating estimation (Modified for Toeplitz)
# =============================
def alternating_estimation(Y, N, P, num_outer=5, num_inner=30, lr_theta=0.05, lr_M=0.01, tol=1e-5):
    dev = Y.device # Get device from input data
    
    # Initialize DOAs via MUSIC
    theta_est = music_initialization(Y, P, N).clone().detach().requires_grad_(True)
    
    # === [Mod 1] Initialize Toeplitz parameters 'c_param' instead of the full M matrix ===
    # We only estimate the first K diagonal parameters (e.g., Bandwidth = 5).
    # This avoids estimating unnecessary distant coupling, resulting in fewer parameters and higher accuracy.
    toeplitz_K = 5 
    
    # Initialization: Set the first element (main diagonal) to 1, others to small values
    c_init = torch.zeros(toeplitz_K, dtype=torch.complex64, device=dev)
    c_init[0] = 1.0 + 0.0j
    for k in range(1, toeplitz_K):
        c_init[k] = 0.05 * (0.5**k) + 0j # Simulate exponential decay
        
    c_param = c_init.clone().detach().requires_grad_(True)
    # ========================================================
    
    R_y = compute_sample_covariance(Y)
    prev_loss = float('inf')

    # Define Helper: Construct and normalize M from the current c_param
    def get_normalized_M(c_p):
        M_raw = build_M_from_toeplitz_params(c_p, N, dev)
        # Enforce M[0,0] = 1 (real) to remove scalar ambiguity
        norm_factor = M_raw[0,0] / torch.abs(M_raw[0,0])
        M_eff = M_raw / norm_factor
        M_eff = M_eff / M_eff[0,0].real
        return M_eff

    for outer in range(num_outer):
        # ------------------------
        # Step 1: Fix M, update theta
        # ------------------------
        optimizer_theta = optim.Adam([theta_est], lr=lr_theta)
        for it in range(num_inner):
            optimizer_theta.zero_grad()
            A_est = steering_vector(N, theta_est)
            
            # Construct M using the current c_param
            M_eff = get_normalized_M(c_param)

            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            loss.backward()
            optimizer_theta.step()
            
            with torch.no_grad():
                theta_est.clamp_(-90.0, 90.0)
        
        # ------------------------
        # Step 2: Fix theta, update M (via c_param)
        # ------------------------
        # === [Mod 2] Optimizer changed to optimize c_param ===
        optimizer_M = optim.Adam([c_param], lr=lr_M) 
        
        for it in range(num_inner):
            optimizer_M.zero_grad()
            A_est = steering_vector(N, theta_est)
            
            # Construct M from c_param (this operation is differentiable)
            M_eff = get_normalized_M(c_param)

            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            loss.backward()
            optimizer_M.step()

        # === Early Stopping Check ===
        with torch.no_grad():
            A_est = steering_vector(N, theta_est)
            M_eff = get_normalized_M(c_param)
            R_model = M_eff @ A_est @ A_est.conj().mT @ M_eff.conj().mT
            curr_loss = torch.mean(torch.abs(R_y - R_model)**2) / torch.mean(torch.abs(R_y)**2)
            
            if abs(prev_loss - curr_loss.item()) < tol:
                break
            prev_loss = curr_loss.item()

    # Finalize
    theta_est, _ = torch.sort(theta_est)
    M_final = get_normalized_M(c_param)

    return theta_est.detach(), M_final.detach()

# =============================
# 8. Run pipeline
# =============================
if __name__ == "__main__":
    Y, true_DOAs, M_true = generate_measurements(N, P, L, SNR_dB, device)
    theta_est, M_est = alternating_estimation(Y, N, P, num_outer=5, num_inner=200)

    print("\nTrue DOAs:", true_DOAs.cpu().numpy())
    print("Estimated DOAs:", theta_est.cpu().numpy())
    error_norm = torch.norm(M_est - M_true) / torch.norm(M_true)
    print("\nRelative Frobenius norm error of M:", error_norm.item())