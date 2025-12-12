import torch
import os
from data.generator import generate_snapshot_sample

# Configuration
NUM_TRAIN_SAMPLES = 5000  # Number of training samples
N = 16
P = 3
L = 128  # Number of snapshots for training (can be the same as testing or more)
SNR_dB = 100  # We usually generate clean data (or high SNR) for training; noise is added during the training process
SAVE_PATH = "dataset/training_dataset.pt"

def generate_training_data():
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
        
    print(f"Generating {NUM_TRAIN_SAMPLES} training samples...")
    device = torch.device("cpu")
    
    X_list = []
    for _ in range(NUM_TRAIN_SAMPLES):
        # We only need X (Clean Signal)
        # randomize=True allows the model to see various angles and MCM (Multi-Channel Measurements)
        X, _, _, _, _ = generate_snapshot_sample(
            N, P, L, SNR_dB, device, randomize=True, use_toeplitz=True
        )
        X_list.append(X)
    
    # Stack into (Num_Samples, N, L)
    Xs = torch.stack(X_list, dim=0)
    
    torch.save(Xs, SAVE_PATH)
    print(f"Training data saved to {SAVE_PATH} | Shape: {Xs.shape}")

if __name__ == "__main__":
    generate_training_data()