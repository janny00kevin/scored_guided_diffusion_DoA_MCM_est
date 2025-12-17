import torch

# continuous beta schedule helpers

# Normally, the standard diffusion model works in discrete steps t = 0, 1, ..., T-1 as
# x_t = \sqrt{\alpha_t} \x_{t-1} + \sqrt{1 - \alpha_t} \bvarepsilon_t.
# This however limits the diffusion steps to T fixed steps, where each step needs to be carefull tuned, 
# and the network only sees discrete noise levels
#
# In this version of the code, we use continuous-time diffusion where a continuous "psuedo-time" variable t \in [0,T] so the forward
# diffusion process is not rewritten as
# dx_t = -\frac{1}{2} \beta(t) \x(t) dt + \sqrt{\beta(t)} dW(t),
# where
# x_t is the noisy signal at continuous time t
# \beta_t is the time-dependent noise schedule
# dW(t) is a Wiener process increment (standard Gaussian distribution)
#
# This way, the noise is added continuously, instead of step-by-step

# This function computes the integral of \beta(s) from 0 to t
# Note that \beta(t) = \beta_\min + (\beta_\max + \beta_\min) * t/T so that its integral is equal to the value that is returned
# in integral_beta_0_to_t
def integral_beta_0_to_t(t, beta_min=1e-4, beta_max=0.05, T=100.0):
    coef = (beta_max - beta_min) / (2.0 * T)
    return beta_min * t + coef * (t ** 2)

# \widebar{\alpha}_t = \exp \left(  -\int_0^t \beta(s) \; ds \right) gives the signal scaling factor at time t so that
# \x_t = \sqrt{ \widebar{\alpha}_t} \x_0 + \sqrt{ 1 - \widebar{\alpha}_t } \bvarepsilon.
# This means the noise schedule is exponential as DDIM requires \widebar{\alpha}_t to be positive and montonic, which is
# what alpha_bar_of_t is returning below
def alpha_bar_of_t(t, beta_min=1e-4, beta_max=0.05, T=100.0):
    return torch.exp(- integral_beta_0_to_t(t, beta_min, beta_max, T))


# Suggested presets to try:
# Conservative (preserve signal longer): beta_min=1e-5, beta_max=0.01, T=50
# Balanced: your current beta_min=1e-4, beta_max=0.02, T=50
# Aggressive (more noising): beta_min=1e-4, beta_max=0.05, T=50
# Long horizon: T=200 with small betas to give smoother sampling.

# \widebar{\alpha}_t is the variance ratio of the original signal component in \x_t​
# \sqrt{\widebar{\alpha}}_t is the signal fraction (how much amplitude of the original signal remains).
# Example: with \widebar{\alpha}_T \approx 0.6050, \sqrt{\widebar{\alpha}}_t \approx 0.7778. That means 
# at the final time a substantial fraction (~78%) of the original signal amplitude remains, i.e., the final 
# state is not pure noise. If you want the final state to be nearly pure noise, make \widebar{\alpha}_T 
# much smaller (e.g., 1e-4–1e-6).