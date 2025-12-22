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
        'color': 'black', 
        'marker': 's', 
        'linestyle': '-', 
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
    "mcm": "M_nmse",
    "x0":  "x0_nmse"
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

def plot_and_save_metric(metric_type, title, ylabel, save_name_base, script_dir):
    """
    Plot and save .mat data for a single metric type (doa, mcm, x0).
    """
    plt.figure(figsize=(8, 6))
    data_found = False

    # For saving data for .mat export
    plot_data_export = {}
    
    # get corresponding data key
    data_key = MAT_KEYS[metric_type]
    
    # read and plot each file
    for label, filename in FILES.items():
        filepath = os.path.join(script_dir, filename)
        data = load_data(filepath)
        
        if data is None:
            continue
            
        if data_key not in data or MAT_KEYS["snr"] not in data:
            if metric_type == 'x0':
                print(f"[Info] '{filename}' does not contain x0 data (likely old version).")
            else:
                print(f"[Warning] Key '{data_key}' not found in {filename}")
            continue

        # flatten arrays
        snrs = data[MAT_KEYS["snr"]].flatten()
        values = data[data_key].flatten()
        
        # plot
        data_found = True
        style = STYLES.get(label, {'linestyle': '-', 'marker': 'o'}) 
        legend_label = style.pop('label', label)
        plt.plot(snrs, values, label=legend_label, **style)

        # prepare data for .mat export
        safe_label = label.replace(" ", "_").replace("-", "_")
        plot_data_export[f"{safe_label}_snr"] = snrs
        plot_data_export[f"{safe_label}_{metric_type}"] = values
        
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
    
    # save png plot
    png_path = os.path.join(script_dir, f"{save_name_base}.png")
    plt.savefig(png_path, dpi=300)
    print(f"[Success] Saved plot image to: test_results/{save_name_base}.png")

    # save .mat data
    mat_path = os.path.join(script_dir, f"{save_name_base}_data.mat")
    sio.savemat(mat_path, plot_data_export)
    print(f"[Success] Saved plot data to : test_results/{save_name_base}_data.mat")
    plt.close()

# =============================
# 3. Main Execution
# =============================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("[Info] Generating plots from configuration...")

    # 1. --- DOA NMSE Plot ---
    plot_and_save_metric(
        metric_type="doa",
        title="DOA Estimation Performance",
        ylabel="DOA NMSE (dB)",
        save_name_base="NMSE_DOA",
        script_dir=script_dir
    )

    # 2. --- MCM NMSE Plot ---
    plot_and_save_metric(
        metric_type="mcm",
        title="MCM Estimation Performance",
        ylabel="MCM NMSE (dB)",
        save_name_base="NMSE_MCM",
        script_dir=script_dir
    )

    plot_and_save_metric(
        metric_type="x0",
        title="Signal Denoising Performance",
        ylabel="Signal NMSE (dB)",
        save_name_base="NMSE_X0_Denoising",
        script_dir=script_dir
    )

if __name__ == "__main__":
    main()