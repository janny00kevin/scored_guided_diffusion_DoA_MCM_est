import torch
import os
from data.generator import generate_snapshot_sample

# =============================
# 1. Configuration
# =============================
# Set default data type to float32
torch.set_default_dtype(torch.float32)

# Use CPU for generation to ensure compatibility when loading on different machines
device = torch.device("cpu")
torch.manual_seed(0)

# Simulation parameters matching your project settings
N = 16            # Number of antennas
P = 3             # Number of sources
L = 128           # Number of snapshots
NUM_TRIALS = 3000 # Number of Monte Carlo trials per SNR level

# List of SNR values (dB) to generate data for
SNR_LEVELS = [-4, -2, 0, 2, 4, 6, 8, 10]

# Directory and Filename for saving
SAVE_DIR = "dataset"
SAVE_FILENAME = "test_data_all_snr.pt"  #

# =============================
# 2. Dataset Generation Function
# =============================
def generate_and_save_dataset(snr_levels, num_trials, save_dir, filename):
    """
    Generates synthetic data for ALL SNR levels and saves it to a SINGLE .pt file.
    The structure will be a dictionary where keys are SNR values.
    """
    
    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")
    else:
        print(f"Saving data to existing directory: {save_dir}")

    print(f"Starting data generation (Number of antennas:{N}, Number of sources:{P}, Number of snapshots:{L})...")

    # Initialize the main dictionary to hold all data
    # Structure: { -4: [sample1, ...], 0: [sample1, ...], ... }
    full_dataset = {} 

    for snr in snr_levels:
        snr_samples = [] # List to hold samples for this specific SNR
        print(f"Generating {num_trials} samples for SNR = {snr} dB...")

        for i in range(num_trials):
            # --------------------------
            # Generate one sample
            # --------------------------
            X, Y, theta_true, M_true, c_true = generate_snapshot_sample(
                N, P, L, snr, device, randomize=True, use_toeplitz=True
            )

            # --------------------------
            # Pack data into a dictionary
            # --------------------------
            sample = {
                "id": i,
                "Y": Y,                  # Observation (N, L)
                "theta_true": theta_true,# Ground truth DOAs (P,)
                "M_true": M_true,        # Ground truth MCM (N, N)
                "snr": snr               # SNR value
            }
            snr_samples.append(sample)
        
        # After finishing one SNR loop, store the list into the main dictionary
        full_dataset[snr] = snr_samples

    # --------------------------
    # Save the CONSOLIDATED file to disk (Outside the loop)
    # --------------------------
    full_path = os.path.join(save_dir, filename)
    torch.save(full_dataset, full_path)
    print(f"\nAll data saved to: {full_path}")
    print("Data generation complete.")

# =============================
# 3. Main Execution
# =============================
if __name__ == "__main__":
    generate_and_save_dataset(SNR_LEVELS, NUM_TRIALS, SAVE_DIR, SAVE_FILENAME)