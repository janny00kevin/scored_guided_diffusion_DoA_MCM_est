import torch
from diffusion.continuous_beta import alpha_bar_of_t
from diffusion.physics_guidance import complex_to_real, complex_stack_from_real, project_x0s_physics

# Vectorized DDIM deterministic guided sampler operating on all L snapshots in parallel

def ddim_epsnet_guided_sampler_batch(y_obs_complex, eps_net, snr,
                                     num_steps=200, T=50.0,
                                     beta_min=1e-4, beta_max=0.02,
                                     guidance_lambda=0.8, device=None,
                                     apply_physics_projection=False):
    device = device or y_obs_complex.device
    eps_net.eval()
    with torch.no_grad():
        Nloc, Lloc = y_obs_complex.shape
        # shape all columns into batch B = L
        y_real = complex_to_real(y_obs_complex.T)  # (L, 2N)
        B = y_real.shape[0]
        t_seq = torch.linspace(T, 0.0, num_steps, device=device)
        x_t = torch.randn_like(y_real, device=device)
        sigma_y2 = (10 ** (-snr / 20.0)) ** 2  # placeholder, user can pass SNR if needed

        for k in range(num_steps - 1):
            t_cur = t_seq[k]
            t_next = t_seq[k+1]
            # batch predict eps
            t_batch = torch.full((B,), t_cur, device=device)
            eps_pred = eps_net(x_t, t_batch)
            # compute alpha bars
            a_bar_cur = alpha_bar_of_t(t_cur, beta_min, beta_max, T)
            a_bar_next = alpha_bar_of_t(t_next, beta_min, beta_max, T)
            sqrt_a_cur = torch.sqrt(a_bar_cur)
            sqrt_1m_a_cur = torch.sqrt(1.0 - a_bar_cur)
            sqrt_a_next = torch.sqrt(a_bar_next)
            sqrt_1m_a_next = torch.sqrt(1.0 - a_bar_next)

            # denoise to x0_hat for entire batch
            x0_hat = (x_t - sqrt_1m_a_cur * eps_pred) / (sqrt_a_cur + 1e-12)

            # guidance in x0 domain using observed y
            grad_x0 = (y_real - x0_hat) / (sigma_y2 + 1e-8)
            x0_hat_guided = x0_hat + guidance_lambda * grad_x0

            # compute eps_guided
            eps_guided = (x_t - sqrt_a_cur * x0_hat_guided) / (sqrt_1m_a_cur + 1e-12)

            # DDIM deterministic update for whole batch
            x_t = sqrt_a_next * x0_hat_guided + sqrt_1m_a_next * eps_guided

            # optional physics projection on batch (operates in real stacked domain)
            if apply_physics_projection and (k % 10 == 0):
                x_t = project_x0s_physics(x_t, enforce_norm=True)

        # final denoised x0 at t_last
        t_last = t_seq[-1]
        t_batch = torch.full((B,), t_last, device=device)
        eps_final = eps_net(x_t, t_batch)
        a_bar_last = alpha_bar_of_t(t_last, beta_min, beta_max, T)
        sqrt_a_last = torch.sqrt(a_bar_last)
        sqrt_1m_a_last = torch.sqrt(1.0 - a_bar_last)
        x0_hat_final = (x_t - sqrt_1m_a_last * eps_final) / (sqrt_a_last + 1e-12)
        # final guidance
        grad_x0 = (y_real - x0_hat_final) / (sigma_y2 + 1e-8)
        x0_hat_final_guided = x0_hat_final + guidance_lambda * grad_x0

        # unstack back to (N, L)
        x0_hat_complex = complex_stack_from_real(x0_hat_final_guided)
        x0_est = x0_hat_complex.T  # (N, L)
        return x0_est