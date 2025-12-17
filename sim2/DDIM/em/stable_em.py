import torch
import torch.optim as optim
#from diffusion.physics_guidance import build_M_from_toeplitz_params if False else None
if False:
    from diffusion.physics_guidance import build_M_from_toeplitz_params
else:
    build_M_from_toeplitz_params = None

# Stable alternating estimation with monotone loss enforcement

def compute_sample_covariance(Y):
    return (Y @ Y.conj().mT) / Y.shape[1]

# construct toeplitz M from first-column c (complex)
def build_M_from_toeplitz_params(c_param, N):
    # c_param: complex vector of length K (first column lags; c_param[0] is diag)
    # N: desired matrix dimension (number of antennas)
    # returns: N x N complex Toeplitz-like mutual coupling matrix.
    K = c_param.shape[0]
    # sanity: K should be <= N (if K>N we only use first N; if K==0 return eye)
    if K <= 0:
        return torch.eye(N, dtype=torch.complex64, device=c_param.device)
    M = torch.eye(N, dtype=torch.complex64, device=c_param.device)
    for k in range(1, min(K, N)):
        v = c_param[k] * torch.ones(N - k, dtype=torch.complex64, device=c_param.device)
        M += torch.diag(v, diagonal=k) + torch.diag(torch.conj(v), diagonal=-k)
    return M


def alternating_estimation_monotone(x0_init, N, P,
                                    num_outer=3, num_inner=100,
                                    lr_theta=5e-2, lr_M=1e-2,
                                    enforce_M11=True, toeplitz_K=None, device=None):
    device = device or x0_init.device
    Y_like = x0_init
    R_y = compute_sample_covariance(Y_like)
    # initialize theta via MUSIC-like (simple grid)
    angles = torch.linspace(-90, 90, 181, device=device)
    from data.generator import steering_vector
    A_grid = steering_vector(N, angles, device=device)
    R_inv = torch.linalg.pinv(R_y)
    spectrum = torch.zeros(angles.shape[0], device=device)
    for i in range(angles.shape[0]):
        a = A_grid[:, i:i+1]
        spectrum[i] = 1.0 / torch.real((a.conj().mT @ R_inv @ a)[0,0])
    _, idx = torch.topk(spectrum, P)
    theta_est = angles[idx].clone().detach().requires_grad_(True)

    # M parameterization
    if toeplitz_K is not None:
        # initialize first-col params (length toeplitz_K)
        K = toeplitz_K
        c_init = torch.zeros(K, dtype=torch.complex64, device=device)
        c_init[0] = 1.0 + 0.0j
        for k in range(1, K):
            c_init[k] = 0.05 * (0.5**k) + 0j
        c_param = c_init.clone().detach().requires_grad_(True)
        optimizer_M = optim.Adam([c_param], lr=lr_M)
        use_toeplitz = True
    else:
        real_init = torch.eye(N, dtype=torch.float32, device=device) + 0.01 * torch.randn(N,N, device=device)
        imag_init = 0.01 * torch.randn(N,N, device=device)
        real_param = real_init.clone().detach().requires_grad_(True)
        imag_param = imag_init.clone().detach().requires_grad_(True)
        optimizer_M = optim.Adam([real_param, imag_param], lr=lr_M)
        use_toeplitz = False

    optimizer_theta = optim.Adam([theta_est], lr=lr_theta)

    # initial loss helpers
    def model_cov(M_est, theta):
        A = steering_vector(N, theta, device=device)
        R_model = M_est @ A @ A.conj().mT @ M_est.conj().mT
        return R_model

    def loss_fn(Ry, R_model):
        return torch.mean(torch.abs(Ry - R_model)**2)

    # helper: build M from current parameters (now passes N when toeplitz)
    def build_M_from_params():
        if use_toeplitz:
            return build_M_from_toeplitz_params(c_param, N)
        else:
            return (real_param + 1j * imag_param)

    for outer in range(num_outer):
        # update theta
        for _ in range(num_inner):
            optimizer_theta.zero_grad()
            M_est = build_M_from_params()
            if enforce_M11:
                M_est = M_est.clone()
                M_est[0,0] = 1.0 + 0j
            R_model = model_cov(M_est, theta_est)
            loss = loss_fn(R_y, R_model)
            loss.backward()
            optimizer_theta.step()
            with torch.no_grad():
                theta_est.clamp_(-90.0, 90.0)

        # update M with backtracking to ensure monotone decrease
        for _ in range(num_inner):
            if use_toeplitz:
                # snapshot current params and loss
                cur_params = c_param.clone().detach()
                cur_loss = loss_fn(R_y, model_cov(build_M_from_toeplitz_params(cur_params, N), theta_est)).item()
                optimizer_M.zero_grad()
                M_est = build_M_from_params()
                if enforce_M11:
                    M_est = M_est.clone(); M_est[0,0]=1.0+0j
                R_model = model_cov(M_est, theta_est)
                loss = loss_fn(R_y, R_model)
                loss.backward()
                optimizer_M.step()
                # evaluate new loss
                new_loss = loss_fn(R_y, model_cov(build_M_from_params(), theta_est)).item()
                if new_loss > cur_loss + 1e-12:
                    # revert and take smaller manual step using stored gradient
                    with torch.no_grad():
                        # revert
                        c_param.copy_(cur_params)
                        # manual small gradient step if grad exists
                        if c_param.grad is not None:
                            # note: after revert grad is stale; recompute gradient using loss at current params if needed
                            # here we do a simple finite fallback: small random perturbation or skip
                            pass
            else:
                cur_real = real_param.clone().detach(); cur_imag = imag_param.clone().detach()
                cur_loss = loss_fn(R_y, model_cov(build_M_from_params(), theta_est)).item()
                optimizer_M.zero_grad()
                M_est = build_M_from_params()
                if enforce_M11:
                    M_est = M_est.clone(); M_est[0,0]=1.0+0j
                R_model = model_cov(M_est, theta_est)
                loss = loss_fn(R_y, R_model)
                loss.backward()
                optimizer_M.step()
                new_loss = loss_fn(R_y, model_cov(build_M_from_params(), theta_est)).item()
                if new_loss > cur_loss + 1e-12:
                    with torch.no_grad():
                        real_param.copy_(cur_real); imag_param.copy_(cur_imag)
                        # manual smaller gradient step using gradient attributes if available
                        if real_param.grad is not None:
                            real_param -= (lr_M * 0.5) * real_param.grad
                        if imag_param.grad is not None:
                            imag_param -= (lr_M * 0.5) * imag_param.grad

    # finalize
    if use_toeplitz:
        M_est_final = build_M_from_toeplitz_params(c_param, N).detach()
        if enforce_M11:
            M_est_final = M_est_final.clone(); M_est_final[0,0] = 1.0+0j
    else:
        M_est_final = (real_param + 1j * imag_param).detach()
        if enforce_M11:
            M_est_final = M_est_final.clone(); M_est_final[0,0] = 1.0+0j

    return torch.sort(theta_est)[0].detach(), M_est_final


# local import to avoid circular import at top
from data.generator import steering_vector
