## `README.md`
```
Modular project for epsilon-net DDIM guided sampling and stable alternating EM for DOA and mutual coupling matrix estimation.
Files:
- models/epsnet_mlp.py
- models/epsnet_unet1d.py
- diffusion/continuous_beta.py
- diffusion/ddim_sampler_parallel.py
- diffusion/physics_guidance.py
- em/stable_em.py
- data/generator.py
- train.py
- main.py

Run `python main.py` to run an end-to-end test (simulates data, trains small epsilon-net, runs DDIM batch sampler, then stable EM).

# Notes
#
# - The modular code aims to vectorize the DDIM guided sampler over all L snapshots (file `diffusion/ddim_sampler_parallel.py`).
# - Physics-informed helpers are in `diffusion/physics_guidance.py` (Toeplitz projection, simple energy normalization).  Use these during sampling via `apply_physics_projection=True`.
# - The EM in `em/stable_em.py` uses backtracking-style reversion if a gradient step increases the loss; it enforces `M[0,0] = 1` at projection steps.
# - This is a starting implementation — tune hyperparameters, SNR-aware `sigma_y2`, and the projection strength for best performance.