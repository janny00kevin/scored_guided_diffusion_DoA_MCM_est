import os
import torch
from data.generator import generate_training_data, generate_testing_data

def get_or_create_training_dataset(num_samples, N, P, L, device, script_dir, use_toeplitz=True):
    dataset_dir = os.path.join(script_dir, "data", "dataset")

    if not os.path.exists(dataset_dir):
        print(f"[Info] Creating directory: {dataset_dir}")
        os.makedirs(dataset_dir)

    filename = f"training_data_Size{num_samples}.pt"
    file_path = os.path.join(dataset_dir, filename)

    if os.path.exists(file_path):
        print(f'[Info] Loading training dataset...')
        Xs_train = torch.load(file_path, map_location=device)
    else:
        print(f"[Info] Generating {num_samples} new training samples...")
        Xs_train = generate_training_data(num_samples, N, P, L, device, use_toeplitz)

    return Xs_train


def get_or_create_testing_dataset(num_samples, N, P, L, snr_levels, device, script_dir, use_toeplitz=True):
    dataset_dir = os.path.join(script_dir, "data", "dataset")

    filename = f"testing_data_Size{num_samples}.pt"
    file_path = os.path.join(dataset_dir, filename)

    if os.path.exists(file_path):
        print(f'[Info] Loading testing dataset...')
        full_dataset = torch.load(file_path, map_location=device)
    else:
        print(f"[Info] Generating {num_samples} new testing samples...")
        full_dataset = generate_testing_data(num_samples, N, P, L, snr_levels, device, use_toeplitz)
        
    return full_dataset