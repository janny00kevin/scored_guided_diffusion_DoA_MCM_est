import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing
# Use file_system strategy to avoid "too many open files" error in multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Pool, cpu_count
import os

# Import the solver
try:
    from MCM_DOA_est_alternately import alternating_estimation
except ImportError:
    print("Error: Could not find 'MCM_DOA_est_alternately.py'.")
    exit()

# ============================================
# Worker Function for Multiprocessing
# ============================================
def process_single_sample(sample_numpy):
    """
    Worker function to process a single sample.
    Input is numpy dict to reduce IPC overhead.
    """
    # Ensure computation happens on CPU
    device = torch.device('cpu')
    
    try:
        # Convert Numpy back to Tensor
        Y = torch.from_numpy(sample_numpy['Y']).to(device)
        theta_true = torch.from_numpy(sample_numpy['theta_true']).to(device)
        M_true = torch.from_numpy(sample_numpy['M_true']).to(device)
        
        N = Y.shape[0]
        P = theta_true.shape[0]
        
        # --- Call the Non-AI Solver ---
        theta_est, M_est = alternating_estimation(Y, N, P, num_outer=5, num_inner=200)
        
        # --- Post-processing ---
        # Sort angles to ensure correct error calculation
        theta_true_sorted, _ = torch.sort(theta_true)
        theta_est_sorted, _ = torch.sort(theta_est)
        
        # Calculate Errors (Scalar)
        # Avoid division by zero by adding epsilon or checking values
        norm_theta = torch.norm(theta_true_sorted).item()
        norm_M = torch.norm(M_true).item()
        
        if norm_theta == 0 or norm_M == 0:
            return None, None

        err_doa = torch.norm(theta_est_sorted - theta_true_sorted).item() / norm_theta
        err_mcm = torch.norm(M_est - M_true).item() / norm_M
        
        # Convert to dB
        # Use a floor of -100dB for perfect estimation (error=0)
        nmse_doa_db = 20 * np.log10(err_doa) if err_doa > 1e-10 else -100.0
        nmse_mcm_db = 20 * np.log10(err_mcm) if err_mcm > 1e-10 else -100.0
        
        return nmse_doa_db, nmse_mcm_db
        
    except Exception as e:
        # In case of solver failure (divergence, etc.)
        # print(f"Sample failed: {e}") # Uncomment for debugging
        return None, None

# ============================================
# Main Execution Function
# ============================================
def run_non_ai_benchmark():
    # Path to the consolidated dataset
    dataset_path = os.path.join("dataset", "test_data_all_snr.pt")
    output_path = os.path.join("test_results", "test_results_nonAI.pt")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset: {dataset_path}")
    full_dataset = torch.load(dataset_path)
    snr_levels = sorted(full_dataset.keys())
    
    # Store results: { snr: {'doa_nmse': val, 'mcm_nmse': val} }
    # We store the averaged values here, but you could store raw lists if needed
    final_results = {
        "snr_levels": snr_levels,
        "doa_nmse_avg": [],
        "mcm_nmse_avg": []
    }

    # Setup Multiprocessing
    # Leave 1-2 cores free for system responsiveness
    num_workers = max(1, cpu_count() - 1)
    print(f"Starting Non-AI benchmark using {num_workers} CPU cores...")

    for snr in snr_levels:
        samples = full_dataset[snr]
        
        # Convert Tensors to Numpy for efficient passing to workers
        samples_numpy = []
        for s in samples:
            samples_numpy.append({
                'Y': s['Y'].numpy(),
                'theta_true': s['theta_true'].numpy(),
                'M_true': s['M_true'].numpy()
            })
            
        print(f"Processing SNR {snr}dB ({len(samples)} trials)...")

        # Run parallel processing
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_sample, samples_numpy), total=len(samples)))
        
        # Filter valid results
        valid_doa = [r[0] for r in results if r is not None and r[0] is not None]
        valid_mcm = [r[1] for r in results if r is not None and r[1] is not None]
        
        # Compute Mean
        avg_doa = np.mean(valid_doa) if valid_doa else None
        avg_mcm = np.mean(valid_mcm) if valid_mcm else None
        
        final_results["doa_nmse_avg"].append(avg_doa)
        final_results["mcm_nmse_avg"].append(avg_mcm)
        
        print(f"  -> Avg DOA NMSE: {avg_doa:.2f} dB")
        print(f"  -> Avg MCM NMSE: {avg_mcm:.2f} dB")

    # Save results to file
    torch.save(final_results, output_path)
    print(f"\nBenchmark complete. Results saved to '{output_path}'.") 

if __name__ == "__main__":
    run_non_ai_benchmark()