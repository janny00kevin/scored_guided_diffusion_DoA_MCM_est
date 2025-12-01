import torch
from models.epsnet_unet1d import EpsNetUNet1D
#from data.generator import make_training_dataset if False else None
if False:
    from data.generator import make_training_dataset
else:
    make_training_dataset = None

# For brevity, we implement a small training runner here; you can call it from main

def train_epsilon_net(Xs, model_type='unet1d', num_epochs=5, batch_size=64, device=None):
    device = device or torch.device('cpu')
    S, Nloc, Lloc = Xs.shape
    dim = 2 * Nloc
    if model_type == 'unet1d':
        from models.epsnet_unet1d import EpsNetUNet1D as Net
    else:
        from models.epsnet_mlp import EpsNetMLP as Net
    net = Net(dim=dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    iters_per_epoch = max(1, (S * Lloc) // batch_size)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for it in range(iters_per_epoch):
            idx_s = torch.randint(0, S, (batch_size,), device=device)
            idx_c = torch.randint(0, Lloc, (batch_size,), device=device)
            x0_batch = torch.stack([Xs[i, :, j] for i, j in zip(idx_s.tolist(), idx_c.tolist())], dim=0).to(device)
            x0_real = torch.cat([x0_batch.real, x0_batch.imag], dim=-1)
            t_cont = torch.rand(batch_size, device=device) * 50.0   # in this case, T = 50
            # simple q_sample continuous
            from diffusion.continuous_beta import alpha_bar_of_t
            a_bar = alpha_bar_of_t(t_cont)
            sqrt_a = torch.sqrt(a_bar).view(-1,1)
            sqrt_1ma = torch.sqrt(1.0 - a_bar).view(-1,1)
            noise = torch.randn_like(x0_real)
            x_t = sqrt_a * x0_real + sqrt_1ma * noise
            pred_eps = net(x_t, t_cont)
            loss = torch.mean((pred_eps - noise)**2)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, loss={total_loss/iters_per_epoch:.6e}")
    return net