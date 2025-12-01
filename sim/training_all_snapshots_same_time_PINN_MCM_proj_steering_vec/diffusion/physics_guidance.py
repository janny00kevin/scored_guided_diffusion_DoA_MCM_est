import torch
import numpy as np

# Physics-informed projections and utilities

def toeplitz_from_first_col(c):
    N = c.shape[0]
    M = torch.zeros((N,N), dtype=torch.complex64, device=c.device)
    for i in range(N):
        for j in range(N):
            idx = j - i
            if idx >= 0:
                M[i,j] = c[idx]
            else:
                M[i,j] = torch.conj(c[-idx])
    return M

def project_M_to_hermitian(M):
    return 0.5 * (M + M.conj().mT)

def enforce_M11_real_one(M):
    M = M.clone()
    M[0,0] = M[0,0].real + 0.j
    M[0,0] = 1.0 + 0.0j
    return M

# Project *batch* of real-domain x0_real (B, 2N) back to complex snapshots, compute covariance and project
# We implement a gentle projection: normalize snapshot energy and optionally apply covariance Toeplitz projection

def complex_stack_from_real(x_real):
    N2 = x_real.shape[-1]
    N = N2 // 2
    return x_real[..., :N] + 1j * x_real[..., N:]

def complex_to_real(x):
    return torch.cat([x.real, x.imag], dim=-1)

def project_x0s_physics(x0s_real, enforce_norm=True, energy_target=None):
    # x0s_real: (B, 2N) stacked real
    x0s_c = complex_stack_from_real(x0s_real)
    if enforce_norm:
        # normalize each snapshot column to match energy_target (or keep relative)
        mags = torch.sqrt(torch.sum(torch.abs(x0s_c)**2, dim=1, keepdim=True))  # (B,1)
        if energy_target is None:
            energy_target = torch.median(mags)
        x0s_c = x0s_c * (energy_target / (mags + 1e-12))
    return complex_to_real(x0s_c)

# project sample covariance to Hermitian Toeplitz (first K lags)
def project_cov_to_toeplitz(R, K=None):
    # R: (N,N) complex
    N = R.shape[0]
    if K is None:
        K = N
    c = [R[i,0] for i in range(N)]
    # average across diagonals
    first_col = torch.zeros(N, dtype=torch.complex64, device=R.device)
    for k in range(N):
        diag_elems = torch.diag(R, diagonal=k)
        first_col[k] = diag_elems.mean()
    return toeplitz_from_first_col(first_col)