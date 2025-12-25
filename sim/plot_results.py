import torch
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import re

# =============================
# 1. Configuration
# =============================
RESULT_DIR = "test_results"

# Mapping: Legend Label -> Filename
# Ensure these filenames match exactly what is in your folder
FILES = {
    # "Non-AI (Unconstrained)":     "test_results_nonAI.pt",
    "Non-AI (Toeplitz)":          "test_results_nonAI_Toe.pt",
    # "Method 2 (DDPM/SDE)":        "test_results_DDPM.pt",
    
    # "DDIM (λ=0.0)":              "test_results_DDIM_lamb0e+00.pt",
    "DDIM (fix $\lambda$=0.4)":    "test_results_DDIM_ori.pt",
    # "DDIM (λ=0.8)":              "test_results_DDIM_lamb8e-01.pt",
    # "DDIM (λ=1.0)":              "test_results_DDIM_lamb1e+00.pt"
    "DDIM (dynamic $\lambda$)" :      "test_results_DDIM_lamb1e+00_bmax2e-02_nmlz_.pt"
}

# Define Plot Styles for consistency
STYLES = {
    # "Non-AI (Unconstrained)":     {'color': 'gray',   'marker': 'o', 'linestyle': '--', 'linewidth': 1.5, 'markersize': 6},
    "Non-AI (Toeplitz)":          {'color': 'blue',   'marker': 's', 'linestyle': '-',  'linewidth': 2,   'markersize': 6},
    # "Method 2 (DDPM/SDE)":        {'color': 'red',    'marker': '^', 'linestyle': '-',  'linewidth': 2,   'markersize': 7},
    
    
    # "DDIM (λ=0.0)":              {'color': 'black',   'marker': 'x', 'linestyle': ':',  'linewidth': 1.5, 'markersize': 6},
    "DDIM (fix $\lambda$=0.4)":    {'color': 'orange',  'marker': '*', 'linestyle': '-',  'linewidth': 2, 'markersize': 9},
    # "DDIM (λ=0.8)":              {'color': 'magenta', 'marker': 'v', 'linestyle': '-',  'linewidth': 2,   'markersize': 6},
    # "DDIM (λ=1.0)":              {'color': 'brown',   'marker': 'p', 'linestyle': '-.', 'linewidth': 2,   'markersize': 6}
    "DDIM (dynamic $\lambda$)" :     {'color': 'green',   'marker': 'D', 'linestyle': '-',  'linewidth': 2,   'markersize': 6}
}

def load_data(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    return torch.load(path)

def sanitize_key(key):
    """將 Label 轉換為 MATLAB 合法的變數名稱 (移除特殊符號)"""
    # 移除 LaTeX 符號、括號、空格，只保留英數字和底線
    clean = re.sub(r'[^a-zA-Z0-9]', '_', key)
    # 移除連續的底線
    clean = re.sub(r'_+', '_', clean)
    # 移除頭尾底線
    return clean.strip('_')

def plot_metric(metric_key, title, ylabel, save_name):
    plt.figure(figsize=(10, 7))
    
    data_found = False
    mat_export_data = {}
    
    for label, filename in FILES.items():
        data = load_data(filename)
        if data is None:
            continue
            
        snrs = data["snr_levels"]
        values = data[metric_key] # e.g., "doa_nmse_avg"
        
        # Filter Data (remove None values)
        clean_snrs = []
        clean_vals = []
        
        for s, v in zip(snrs, values):
            # Sometimes initialization fails result in None
            # We also filter out extremely low dummy values like -100 if you want detailed view
            if v is not None: 
                clean_snrs.append(s)
                clean_vals.append(v)
        
        if clean_snrs:
            data_found = True
            style = STYLES.get(label, {})
            plt.plot(clean_snrs, clean_vals, label=label, **style)

            safe_label = sanitize_key(label)
            mat_export_data[f"{safe_label}_x"] = clean_snrs
            mat_export_data[f"{safe_label}_y"] = clean_vals

    if not data_found:
        print(f"No valid data found for {title}. Skipping plot.")
        plt.close()
        return

    # Graph Styling
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.legend(fontsize=12, loc='best')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Save Plot
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    save_path = os.path.join(RESULT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")

    mat_name = save_name.replace('.png', '.mat')
    save_path_mat = os.path.join(RESULT_DIR, mat_name)
    sio.savemat(save_path_mat, mat_export_data)
    print(f"Saved data for Matlab to: {save_path_mat}")

    plt.close()

def main():
    print("Generating comprehensive comparison plots...")

    # 1. Compare DOA NMSE
    plot_metric(
        metric_key="doa_nmse_avg",
        title="DOA Estimation Performance",
        ylabel="DOA NMSE (dB)",
        save_name="comparison_all_doa.png"
    )

    # 2. Compare MCM NMSE
    plot_metric(
        metric_key="mcm_nmse_avg",
        title="MCM Estimation Performance",
        ylabel="MCM NMSE (dB)",
        save_name="comparison_all_mcm.png"
    )

    print("Done!")

if __name__ == "__main__":
    main()