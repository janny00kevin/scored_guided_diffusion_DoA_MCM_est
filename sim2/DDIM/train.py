import torch
import os
import copy
from diffusion.continuous_beta import alpha_bar_of_t

def train_epsilon_net(Xs, model_type='unet1d', num_epochs=5, batch_size=64, lr=1e-3,
                      beta_min=1e-4, beta_max=0.02, T=50,
                      val_split=0.1, patience=15, # Early stopping params
                      device=None, script_dir=None, model_file_name=None):
    if model_type == 'mlp':
        from diffusion.physics_guidance import complex_to_real_concat as complex_to_real
    elif model_type == 'unet1d':
        from diffusion.physics_guidance import complex_to_real_stack as complex_to_real
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    device = device or torch.device('cpu')
    S, Nloc, Lloc = Xs.shape  # num of samples, num of antenna, num of snapshots L
    dim = 2 * Nloc  # concat the real and imaginary part

    # --- 1. Data Preparation ---
    # (S, N, L) -> (S, L, N) -> (S*L, N) -> (S*L, 2, N) for parallelly training
    Xs_flat = Xs.permute(0, 2, 1).reshape(-1, Nloc) # Xs_flat: (S*L, N)
    Xs_real = complex_to_real(Xs_flat)  # (S*L, 2, N)
    
    # Split into Train and Validation
    num_total = Xs_real.shape[0]
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    # Shuffle before splitting
    indices = torch.randperm(num_total, device=device)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    train_data = Xs_real[train_idx]
    val_data = Xs_real[val_idx]

    # Compute normalization stats on TRAINING data only
    data_mean = torch.mean(train_data)
    data_std = torch.std(train_data)
    if data_std < 1e-8: data_std = torch.tensor(1.0, device=device)

    # Apply normalization
    train_data_norm = (train_data - data_mean) / data_std
    val_data_norm = (val_data - data_mean) / data_std
    print(f"[Info] Training Samples: {num_train} | Validation Samples: {num_val}")

    # --- 2. Model & Optimizer Setup ---
    if model_type == 'unet1d':
        from models.epsnet_unet1d import EpsNetUNet1D as Net
        net = Net(dim=dim).to(device)
    elif model_type == 'mlp':
        from models.epsnet_mlp import EpsNetMLP as Net
        net = Net(dim=dim, hidden=1024, time_emb_dim=128).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Learning Rate Scheduler: Reduce LR if val_loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Early Stopping Variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    best_epoch = 0

    # --- 3. Training Loop ---
    iters_per_epoch = max(1, num_train // batch_size)
    for epoch in range(num_epochs):
        # --- A. Train Step ---
        net.train()
        total_train_loss = 0.0
        train_indices = torch.randperm(num_train, device=device) # eq. to shuffling
        for i in range(iters_per_epoch):
            # draw the indices for the batch and use it to draw the flatten data
            start = i * batch_size
            end = start + batch_size
            batch_idx = train_indices[start:end]
            x0_batch = train_data_norm[batch_idx]  # (Batch size, 2N)

            # randomly pick a time step \x_t to train
            t_cont = torch.rand(x0_batch.shape[0], device=device) * T   # in this case, T = 50

            # simple q_sample continuous
            a_bar = alpha_bar_of_t(t_cont, beta_min, beta_max, T)
            if model_type == 'mlp':
                sqrt_a = torch.sqrt(a_bar).view(-1, 1)
                sqrt_1ma = torch.sqrt(1.0 - a_bar).view(-1, 1)
            elif model_type == 'unet1d':
                sqrt_a = torch.sqrt(a_bar).view(-1, 1, 1)
                sqrt_1ma = torch.sqrt(1.0 - a_bar).view(-1, 1, 1)

            noise = torch.randn_like(x0_batch)
            x_t = sqrt_a * x0_batch + sqrt_1ma * noise

            pred_eps = net(x_t, t_cont)
            loss = torch.mean((pred_eps - noise)**2)

            opt.zero_grad(); loss.backward(); opt.step()
            total_train_loss += float(loss.item())
        avg_train_loss = total_train_loss / iters_per_epoch

        # --- B. Validation Step (No Grad) ---
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            # Process validation set in chunks to avoid OOM
            val_iters = max(1, num_val // batch_size)
            for i in range(val_iters):
                start = i * batch_size
                end = min(start + batch_size, num_val)
                x0_val = val_data_norm[start:end]
                
                t_cont_val = torch.rand(x0_val.shape[0], device=device) * T
                a_bar_val = alpha_bar_of_t(t_cont_val, beta_min, beta_max, T)
                # x_t_val = torch.sqrt(a_bar_val).view(-1, 1, 1) * x0_val + \
                #           torch.sqrt(1.0 - a_bar_val).view(-1, 1, 1) * torch.randn_like(x0_val)
                
                # Note: We calculate loss against the ADDED noise, but since we generated it
                # implicitly above, let's regenerate explicitly for loss calculation
                noise_val = torch.randn_like(x0_val)
                if model_type == 'mlp':
                    x_t_val = torch.sqrt(a_bar_val).view(-1, 1) * x0_val + \
                              torch.sqrt(1.0 - a_bar_val).view(-1, 1) * noise_val
                elif model_type == 'unet1d':
                    x_t_val = torch.sqrt(a_bar_val).view(-1, 1, 1) * x0_val + \
                            torch.sqrt(1.0 - a_bar_val).view(-1, 1, 1) * noise_val
                
                pred_val = net(x_t_val, t_cont_val)
                val_loss = torch.mean((pred_val - noise_val)**2)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / val_iters

        # --- C. Logging & Updates ---
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}", end="")
        # Update Scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = copy.deepcopy(net.state_dict()) # Keep best in memory
            best_epoch = epoch + 1  # [NEW] Record the epoch number
            print("  <-- New Best Model")
        else:
            early_stop_counter += 1
            print("")
            if early_stop_counter >= patience:
                print(f"[Info] Early stopping triggered at epoch {epoch+1}")
                break

    # --- 4. Finalize & Save ---
    print(f"Training finished. Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}")
    
    # Load the best weights back into the model before returning
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        if script_dir:
            dataset_dir = os.path.join(script_dir, "weights")
            if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
            
            file_name = model_file_name
            save_path = os.path.join(dataset_dir, file_name)
            
            checkpoint = {
                'model_state_dict': best_model_state,
                'config': {'T': T, 'beta_min': beta_min, 'beta_max': beta_max},
                'data_mean': data_mean,
                'data_std': data_std,
                'epoch': best_epoch,  # Use the tracked best_epoch
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved to {save_path}")

    return net