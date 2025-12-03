import torch
import matplotlib.pyplot as plt
import os

# =============================
# 1. Configuration
# =============================
RESULT_DIR = "test_results"
FILES = {
    "Non-AI (Unconstrained)": "test_results_nonAI.pt",
    "Non-AI (Toeplitz)": "test_results_nonAI_Toe.pt"
}

def load_data(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    return torch.load(path)

def plot_metric(metric_key, title, ylabel, save_name):
    plt.figure(figsize=(10, 7))
    
    # Define styles for different curves
    styles = {
        "Non-AI (Unconstrained)": {'marker': 'o', 'linestyle': '--', 'color': 'gray'},
        "Non-AI (Toeplitz)": {'marker': 's', 'linestyle': '-', 'color': 'blue'}
    }

    data_found = False
    
    for label, filename in FILES.items():
        data = load_data(filename)
        if data is None:
            continue
            
        snrs = data["snr_levels"]
        values = data[metric_key]
        
        # Filter out None values
        clean_snrs = []
        clean_vals = []
        for s, v in zip(snrs, values):
            if v is not None:
                clean_snrs.append(s)
                clean_vals.append(v)
        
        if clean_snrs:
            data_found = True
            style = styles.get(label, {})
            plt.plot(clean_snrs, clean_vals, label=label, linewidth=2, **style)

    if not data_found:
        print(f"No valid data found for {title}. Skipping plot.")
        plt.close()
        return

    plt.title(title, fontsize=16)
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Save plot
    save_path = os.path.join(RESULT_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison plot to: {save_path}")
    plt.close()

def main():
    if not os.path.exists(RESULT_DIR):
        print(f"Error: Directory '{RESULT_DIR}' not found.")
        return

    print("Generating comparison plots...")

    # 1. Compare DOA NMSE
    plot_metric(
        metric_key="doa_nmse_avg",
        title="DOA Estimation Performance Comparison",
        ylabel="DOA NMSE (dB)",
        save_name="comparison_doa_nmse.png"
    )

    # 2. Compare MCM NMSE
    plot_metric(
        metric_key="mcm_nmse_avg",
        title="MCM Estimation Performance Comparison",
        ylabel="MCM NMSE (dB)",
        save_name="comparison_mcm_nmse.png"
    )

    print("Done!")

if __name__ == "__main__":
    main()