import torch
import math
import os

def steering_vector(N, theta_deg, device=None):
    dev = device or torch.device('cpu')
    theta = torch.as_tensor(theta_deg, dtype=torch.float32, device=dev)
    theta_rad = theta * (math.pi / 180.0)
    d = 0.5
    k = 2.0 * math.pi * d * torch.sin(theta_rad)
    n = torch.arange(0, N, dtype=torch.float32, device=dev)
    phase = -1j * (n[:, None] * k[None, :])
    return torch.exp(phase)

# We assume the MCM is Toeplitz first, with the first element equals to 1
def generate_mutual_coupling_matrix_random_toeplitz(N, max_band=3, device=None):
    dev = device or torch.device('cpu')
    c0 = 1.0 + 0.0j
    c = [torch.tensor(c0, dtype=torch.complex64, device=dev)]
    for k in range(1, max_band):
        mag = float(0.1 * (0.6 ** k))
        phase = (torch.rand(1).item() - 0.5) * math.pi / 3.0
        val = mag * (math.cos(phase) + 1j * math.sin(phase))
        c.append(torch.tensor(val, dtype=torch.complex64, device=dev))
    c = torch.stack(c)
    M = torch.eye(N, dtype=torch.complex64, device=dev)
    for k in range(1, c.shape[0]):
        diag = c[k] * torch.ones(N - k, dtype=torch.complex64, device=dev)
        M += torch.diag(diag, diagonal=k) + torch.diag(torch.conj(diag), diagonal=-k)
    return M, c

# Generate L snapshots for better accuraacy
def generate_snapshot_sample(N, P, L, SNR_dB, device, use_toeplitz=True):
    dev = device or torch.device('cpu')
    
    thetas = []
    low, high = -60.0, 60.0
    for p in range(P):
        while True:
            cand = (torch.rand(1).item()) * (high - low) + low
            if all(abs(cand - t) > 5.0 for t in thetas):
                thetas.append(cand); break
    theta_true = torch.tensor(thetas[:P], dtype=torch.float32, device=dev)

    A = steering_vector(N, theta_true, device=dev)
    if use_toeplitz:
        M_true, c_true = generate_mutual_coupling_matrix_random_toeplitz(N, max_band=4, device=dev)
    else:
        rand_mat = (0.05 * (torch.randn(N, N, device=dev) + 1j * torch.randn(N, N, device=dev))/math.sqrt(2))
        M_true = torch.eye(N, dtype=torch.complex64, device=dev) + rand_mat
        M_true = 0.5 * (M_true + M_true.conj().mT)
        c_true = None
    S = (torch.randn(P, L, dtype=torch.float32, device=dev) + 1j * torch.randn(P, L, device=dev)) / math.sqrt(2)
    X = M_true @ (A @ S)
    sigma_n = 10 ** (-SNR_dB / 20)
    noise = sigma_n * (torch.randn(N, L, dtype=torch.float32, device=dev) + 1j * torch.randn(N, L, device=dev)) / math.sqrt(2)
    Y = X + noise
    return X, Y, theta_true, M_true, c_true

def generate_training_data(num_train_samples, N, P, L, device, use_toeplitz=True):
    # The training_dataset only contains the Clean Signal \x to train the espilon net
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    # print(f"[Info] Generating {num_train_samples} training samples...")
    device = torch.device("cpu")
    
    X_list = []
    for _ in range(num_train_samples):
        # We only need X (Clean Signal), SNR_dB doesn't matter here
        X, _, _, _, _ = generate_snapshot_sample(N, P, L, SNR_dB=0, 
                                                 device=device, use_toeplitz=use_toeplitz)
        X_list.append(X)
    
    # Stack into (Num_Samples, N, L)
    Xs = torch.stack(X_list, dim=0)
    
    file_name = f"training_data_Size{num_train_samples:.0f}.pt"
    save_path = os.path.join(dataset_dir, file_name)

    torch.save(Xs, save_path)
    print(f"[Info] Training data saved. | Shape: {Xs.shape}")

    return Xs


def generate_testing_data(num_samples, N, P, L, snr_levels, device, use_toeplitz=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Generating num_samples clean \X, \C_R, angles
    list_X = []
    list_theta = []
    list_M = []

    for _ in range(num_samples):
        X_clean, _, theta_true, M_true, _ = generate_snapshot_sample(
            N, P, L, SNR_dB=0, device=device, use_toeplitz=use_toeplitz)
        
        list_X.append(X_clean.cpu())
        list_theta.append(theta_true.cpu())
        list_M.append(M_true.cpu())

    # X_master shape: (num_samples, N, L)
    Xs = torch.stack(list_X, dim=0)
    thetas = torch.stack(list_theta, dim=0)
    Ms = torch.stack(list_M, dim=0)
    # -----------------------------
    
    # For each SNR level, generate noisy observations
    observations = {}
    Xs = Xs.to(device)
    for snr in snr_levels:
        # print(f"[Info] Vectorized calculating for SNR = {snr} dB ...")
        sigma_n = 10 ** (-snr / 20)
        
        # (num_samples, N, L)
        noise = sigma_n * (torch.randn_like(Xs) + 1j * torch.randn_like(Xs)) / math.sqrt(2)

        Ys = Xs + noise
        
        observations[snr] = Ys.cpu()
    # -----------------------------

    # Saving
    full_dataset = {
        "config": {"N": N, "P": P, "L": L, "snr_levels": snr_levels, "num_samples": num_samples},
        # clean data (Tensors) (num_samples, N, L)
        "X_clean": Xs,
        "theta_true": thetas,
        "M_true": Ms,
        # Observations (Dict of Tensors) (num_snr_levels, num_samples, N, L)
        "observations": observations 
    }
    
    file_name = f"testing_data_Size{num_samples:.0f}.pt"
    save_path = os.path.join(dataset_dir, file_name)

    torch.save(full_dataset, save_path)
    print(f"[Info] Testing dataset saved.")

    return full_dataset







    # full_dataset = {} 
    # for snr in snr_levels:
    #     snr_samples = [] # List to hold samples for this specific SNR
    #     # print(f"[Info] Generating {num_testing_samples} samples for SNR = {snr} dB...")

    #     for i in range(num_samples):
    #         # --------------------------
    #         # Generate one sample
    #         # --------------------------
    #         X, Y, theta_true, M_true, c_true = generate_snapshot_sample(
    #             N, P, L, snr, device, randomize=True, use_toeplitz=True
    #         )

    #         # --------------------------
    #         # Pack data into a dictionary
    #         # --------------------------
    #         sample = {
    #             "id": i,
    #             "Y": Y,                  # Observation (N, L)
    #             "theta_true": theta_true,# Ground truth DOAs (P,)
    #             "M_true": M_true,        # Ground truth MCM (N, N)
    #             "snr": snr               # SNR value
    #         }
    #         snr_samples.append(sample)
        
    #     # After finishing one SNR loop, store the list into the main dictionary
    #     full_dataset[snr] = snr_samples


    # return

