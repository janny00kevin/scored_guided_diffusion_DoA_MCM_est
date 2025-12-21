import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np

# =============================
# 1. Configuration
# =============================

# load .mat files to plot
FILES = {
    "Non-AI": "NMSE_Baseline_non_AI.mat",
    "DDIM":   "NMSE_DDIM_ep50_lr1e-04_t1000_bmax2e-02.mat"
}

# 'g-o'  -> color: green, linestyle: -,  marker: o
# 'b--s' -> color: blue,  linestyle: --, marker: s
STYLES = {
    "Non-AI": {
        'color': 'blue', 
        'marker': 's', 
        'linestyle': '--', 
        'linewidth': 2, 
        'markersize': 8
    },
    "DDIM": {
        'color': 'green', 
        'marker': 'o', 
        'linestyle': '-',  
        'linewidth': 2, 
        'markersize': 8
    }
}

# .mat keys mapping
MAT_KEYS = {
    "snr": "snr_range",
    "doa": "theta_nmse",
    "mcm": "M_nmse"
}

# =============================
# 2. Helper Functions
# =============================
def load_data(filepath):
    """Load .mat file"""
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return None
    try:
        return sio.loadmat(filepath)
    except Exception as e:
        print(f"[Error] Failed to load {filepath}: {e}")
        return None

def plot_single_metric(metric_type, title, ylabel, save_name, script_dir):
    """
    metric_type: 'doa' or 'mcm'
    """
    plt.figure(figsize=(8, 6))
    data_found = False
    
    # get corresponding data key
    data_key = MAT_KEYS[metric_type]
    
    # read and plot each file
    for label, filename in FILES.items():
        filepath = os.path.join(script_dir, filename)
        data = load_data(filepath)
        
        if data is None:
            continue
            
        if data_key not in data or MAT_KEYS["snr"] not in data:
            print(f"[Warning] Key '{data_key}' or SNR key not found in {filename}")
            continue

        # flatten arrays
        snrs = data[MAT_KEYS["snr"]].flatten()
        values = data[data_key].flatten()
        
        # plot
        data_found = True
        style = STYLES.get(label, {'linestyle': '-', 'marker': 'o'}) # 預設樣式
        plt.plot(snrs, values, label=label, **style)
        
        # use the same x-ticks (SNR dB) for all plots
        last_valid_snrs = snrs

    if not data_found:
        print(f"[Info] No valid data found for {title}. Skipping plot.")
        plt.close()
        return

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    if 'last_valid_snrs' in locals():
        plt.xticks(last_valid_snrs)
        
    plt.tight_layout()
    
    # save plot
    save_path = os.path.join(script_dir, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Saved plot to: {save_name}")
    plt.close()

# =============================
# 3. Main Execution
# =============================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("[Info] Generating plots from configuration...")

    # 1. --- DOA NMSE Plot ---
    plot_single_metric(
        metric_type="doa",
        title="DOA Estimation Performance",
        ylabel="DOA NMSE (dB)",
        save_name="NMSE_DOA.png",
        script_dir=script_dir
    )

    # 2. --- MCM NMSE Plot ---
    plot_single_metric(
        metric_type="mcm",
        title="MCM Estimation Performance",
        ylabel="MCM NMSE (dB)",
        save_name="NMSE_MCM.png",
        script_dir=script_dir
    )

if __name__ == "__main__":
    main()