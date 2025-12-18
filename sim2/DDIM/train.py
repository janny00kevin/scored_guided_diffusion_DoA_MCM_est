import torch
import os
from diffusion.continuous_beta import alpha_bar_of_t

# For brevity, we implement a small training runner here; you can call it from main

def train_epsilon_net(Xs, model_type='unet1d', num_epochs=5, batch_size=64, lr=1e-3,
                      beta_min=1e-4, beta_max=0.02, T=50, device=None, script_dir=None):
    device = device or torch.device('cpu')
    S, Nloc, Lloc = Xs.shape  # num of samples, num of antenna, num of snapshots L
    dim = 2 * Nloc  # concat the real and imaginary part

    # (S, N, L) -> (S, L, N) -> (S*L, N) -> (S*L, 2N) for parallelly training
    Xs_flat = Xs.permute(0, 2, 1).reshape(-1, Nloc)
    Xs_real = torch.cat([Xs_flat.real, Xs_flat.imag], dim=-1).to(device)
    num_total_snapshots = Xs_real.shape[0]

    # NN model init
    if model_type == 'unet1d':
        from models.epsnet_unet1d import EpsNetUNet1D as Net
        net = Net(dim=dim).to(device)
    else:
        from models.epsnet_mlp import EpsNetMLP as Net
        net = Net(dim=dim, hidden=1024, time_emb_dim=128).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    iters_per_epoch = max(1, num_total_snapshots // batch_size)
    for epoch in range(num_epochs):
        total_loss = 0.0
        indices = torch.randperm(num_total_snapshots, device=device) # eq. to shuffling
        for i in range(iters_per_epoch):
            # draw the indices for the batch and use it to draw the flatten data
            start = i * batch_size
            end = start + batch_size
            batch_idx = indices[start:end]
            x0_batch_real = Xs_real[batch_idx]  # (Batch size, 2N)

            # randomly pick a time step \x_t to train
            t_cont = torch.rand(batch_size, device=device) * T   # in this case, T = 50

            # simple q_sample continuous
            a_bar = alpha_bar_of_t(t_cont, beta_min, beta_max, T)
            sqrt_a = torch.sqrt(a_bar).view(-1,1)
            sqrt_1ma = torch.sqrt(1.0 - a_bar).view(-1,1)

            noise = torch.randn_like(x0_batch_real)
            x_t = sqrt_a * x0_batch_real + sqrt_1ma * noise

            pred_eps = net(x_t, t_cont)
            loss = torch.mean((pred_eps - noise)**2)

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, loss={total_loss/iters_per_epoch:.6f}")
    
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'config': {
            'T': T,
            'beta_min': beta_min,
            'beta_max': beta_max
        }
    }
    file_name = f"DDIM_ep{epoch}_lr{lr:.0e}_t{int(T)}_bmax{beta_max:.0e}.pth"
    dataset_dir = os.path.join(script_dir, "weights")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    save_path = os.path.join(dataset_dir, file_name)
    torch.save(checkpoint, save_path)
    print(f"Model and statistics saved to {save_path}")
        
    return net