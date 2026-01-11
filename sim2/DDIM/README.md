# Score-Guided Diffusion for DOA and MCM Estimation

This repository implements a score-guided diffusion framework (DDIM) for joint Direction of Arrival (DOA) and Mutual Coupling Matrix (MCM) estimation.

## Core Methods
1. **Non-AI**: Baseline alternating EM optimization without neural denoising.
2. **DDIM (w/o CL)**: Standard DDIM guided sampling using a supervised MLP/UNet.
3. **DDIM (w/ CL)**: DDIM guided sampling using an MLP trained with Contrastive Learning (InfoNCE) for noise-invariant feature extraction.

## Repository Structure
- `main_contrastive_learning.py`: Entry point for CL-based training and testing.
- `main_non_CL.py`: Entry point for standard supervised training and testing.
- `test_baseline_non_AI.py`: Script to run the non-AI optimization baseline.
- `train.py`: Contains training loops for both standard and contrastive loss.
- `models/`: Architecture definitions (`epsnet_mlp.py`, `epsnet_unet1d.py`).
- `diffusion/`: DDIM sampling logic and continuous noise schedules.
- `em/`: Batch-parallelized alternating EM algorithm.
- `data/`: Dataset generation and loading utilities.
- `test_results/`: 
    - `NMSE_raw_mats/`: Storage for `.mat` result files.
    - `NMSE_plot_png/`: Exported performance curves.
    - `NMSE_plot.py`: Aggregator script to plot results from different methods.

## Workflow

### 1. Training
Set `RUN_ID = 1` in `main_contrastive_learning.py` or `main_non_CL.py`.
- **Process**: Generates training data (if missing) and trains the epsilon-net.
- **Output**: Model checkpoints (weights, mean, std) are saved in the `weights/` directory.

### 2. Testing
Set `RUN_ID = 2` in the respective main scripts (`main_non_CL.py` or `main_contrastive_learning.py`).

* **Process Details**:
    * **Model Loading**: Retrieves the best-trained weights and normalization statistics from the `weights/`.
    * **DDIM Parallel Denoising**: Performs deterministic reverse diffusion to recover the clean signal $\hat{x}_0$. This is implemented in `diffusion/ddim_sampler_parallel.py`, which supports vectorized sampling and incorporates physical guidance.
    * **Batch EM Estimation**: The denoised $\hat{x}_0$ is passed to the EM algorithm in `em/stable_em_batch.py`. This script parallelizes alternating minimization to jointly estimate DOAs (initialized by batch MUSIC) and the MCM (assumed Toeplitz).
    * **Metric Accumulation**: The `NMSEAccumulator` class in `test_results/NMSE_calculation.py` tracks the error for $x_0$, DOA, and MCM across all mini-batches.

* **Output**: 
    * Final averaged NMSE results for each SNR level are saved as `.mat` files in `test_results/NMSE_raw_mats/`.

### 3. Visualization
Run `python test_results/NMSE_plot.py`.
- This script reads the `.mat` files generated during testing and produces comparison plots in `test_results/NMSE_plot_png/`.