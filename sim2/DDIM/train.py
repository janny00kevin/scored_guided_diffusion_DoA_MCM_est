import torch
import torch.nn.functional as F
import os
import copy
from diffusion.continuous_beta import alpha_bar_of_t
from diffusion.physics_guidance import complex_to_real_concat as complex_to_real

def info_nce_loss(features_a, features_b, temperature=0.1):
    """
    Calculates InfoNCE loss.
    features_a: (Batch, dim) - Embeddings from View A
    features_b: (Batch, dim) - Embeddings from View B
    """
    # Normalize features (Cosine Similarity)
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)

    # Similarity matrix: (Batch, Batch)
    # logits[i, j] = similarity between sample i (view A) and sample j (view B)
    # Diagonal = Positive Pairs (Same Scenario)
    # Off-Diagonal = Negative Pairs (Different Scenarios)
    logits = torch.matmul(features_a, features_b.T) / temperature

    # Labels: The positive pair for i is i
    labels = torch.arange(logits.shape[0], device=logits.device)

    return F.cross_entropy(logits, labels)

def train_epsilon_net(Xs, model_type='mlp', num_epochs=5, batch_size=64, lr=1e-3,
                      beta_min=1e-4, beta_max=0.02, T=50,
                      val_split=0.1, patience=15, # Early stopping params
                      device=None, script_dir=None, model_file_name=None,
                      contrastive_weight=0.1, temperature=0.1): # [NEW] Weight for InfoNCE loss
    if model_type != 'mlp':
        raise NotImplementedError("This InfoNCE implementation is optimized for 'mlp'.")
    
    device = device or torch.device('cpu')
    
    S, Nloc, Lloc = Xs.shape # Xs: (S, N, L) -> Scenarios, Antennas, Snapshots
    dim = 2 * Nloc 

    # --- 1. Data Preparation (Structured) ---
    # We must preserve the 'S' dimension to sample views from the same scenario.
    # Permute to (S, L, N) then Concatenate Real/Imag -> (S, L, 2N)
    Xs_perm = Xs.permute(0, 2, 1)
    Xs_real = complex_to_real(Xs_perm)

    # Split Scenarios (S) into Train and Validation
    # Important: We split by S, not S*L, to ensure validation tests unseen channels.
    num_val_s = int(S * val_split)
    num_train_s = S - num_val_s

    indices = torch.randperm(S, device=device)
    train_idx, val_idx = indices[:num_train_s], indices[num_train_s:]

    train_data = Xs_real[train_idx] # Shape: (Train_S, L, 2N)
    val_data = Xs_real[val_idx]     # Shape: (Val_S, L, 2N)

    # Compute stats on TRAINING data
    data_mean = torch.mean(train_data)
    data_std = torch.std(train_data)
    if data_std < 1e-8: data_std = torch.tensor(1.0, device=device)

    # Normalize
    train_data_norm = (train_data - data_mean) / data_std
    val_data_norm = (val_data - data_mean) / data_std
    
    # Flatten Validation Data for standard evaluation (Val_S * L, 2N)
    val_data_flat = val_data_norm.reshape(-1, dim)
    num_val_total = val_data_flat.shape[0]

    print(f"[Info] Training Scenarios: {num_train_s} (x{Lloc} snapshots) | Validation Snapshots: {num_val_total}")

    # --- 2. Model & Optimizer Setup ---
    from models.epsnet_mlp import EpsNetMLP as Net
    net = Net(dim=dim, hidden=1024, time_emb_dim=128).to(device)
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=200, verbose=False
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    best_epoch = 0

    # --- 3. Training Loop ---
    # We iterate over Scenarios (num_train_s), processing a batch of scenarios at a time.
    iters_per_epoch = max(1, num_train_s // batch_size)
    for epoch in range(num_epochs):
        # --- A. Train Step ---
        net.train()
        total_train_loss = 0.0
        total_mse = 0.0
        total_nce = 0.0
        
        # Shuffle scenarios every epoch
        scenario_indices = torch.randperm(num_train_s, device=device)

        for i in range(iters_per_epoch):
            start = i * batch_size
            end = start + batch_size
            batch_scenarios = scenario_indices[start:end]
            current_bs = batch_scenarios.shape[0]

            # [AUGMENTATION] Select two random snapshots for EACH scenario in batch
            # idx_1, idx_2: (Batch,)
            idx_1 = torch.randint(0, Lloc, (current_bs,), device=device)
            # Ensure idx_2 is different from idx_1 (simple shift)
            idx_2 = (idx_1 + torch.randint(1, Lloc, (current_bs,), device=device)) % Lloc

            # Extract Views: train_data_norm is (Train_S, L, 2N)
            x0_view1 = train_data_norm[batch_scenarios, idx_1, :] # (Batch, 2N)
            x0_view2 = train_data_norm[batch_scenarios, idx_2, :] # (Batch, 2N)

            # Sample t (Same t for both views is standard for contrastive diffusion)
            t_cont = torch.rand(current_bs, device=device) * T
            
            # Get noise schedule
            a_bar = alpha_bar_of_t(t_cont, beta_min, beta_max, T).view(-1, 1)
            sqrt_a = torch.sqrt(a_bar)
            sqrt_1ma = torch.sqrt(1.0 - a_bar)

            # Add Noise
            eps_1 = torch.randn_like(x0_view1)
            eps_2 = torch.randn_like(x0_view2)

            x_t_1 = sqrt_a * x0_view1 + sqrt_1ma * eps_1
            x_t_2 = sqrt_a * x0_view2 + sqrt_1ma * eps_2

            # Forward Pass (Get Embedding)
            pred_eps_1, z_1 = net(x_t_1, t_cont, return_embedding=True)
            pred_eps_2, z_2 = net(x_t_2, t_cont, return_embedding=True)

            # Calculate Losses
            # 1. MSE (Reconstruction) - Average of both views
            loss_mse = 0.5 * (torch.mean((pred_eps_1 - eps_1)**2) + torch.mean((pred_eps_2 - eps_2)**2))
            
            # 2. InfoNCE (Contrastive)
            loss_nce = info_nce_loss(z_1, z_2, temperature)

            # Total
            loss = loss_mse + contrastive_weight * loss_nce

            opt.zero_grad(); loss.backward(); opt.step()
            
            total_train_loss += loss.item()
            total_mse += loss_mse.item()
            total_nce += loss_nce.item()

        avg_train_loss = total_train_loss / iters_per_epoch
        avg_mse = total_mse / iters_per_epoch
        avg_nce = total_nce / iters_per_epoch

        # --- B. Validation Step (Standard MSE) ---
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            # Standard validation on flattened snapshots (like original)
            val_iters = max(1, num_val_total // (batch_size * 2)) # Larger batch for eval
            
            for i in range(val_iters):
                start = i * (batch_size * 2)
                end = min(start + (batch_size * 2), num_val_total)
                x0_val = val_data_flat[start:end]
                
                t_cont_val = torch.rand(x0_val.shape[0], device=device) * T
                a_bar_val = alpha_bar_of_t(t_cont_val, beta_min, beta_max, T).view(-1, 1)
                
                noise_val = torch.randn_like(x0_val)
                x_t_val = torch.sqrt(a_bar_val) * x0_val + torch.sqrt(1.0 - a_bar_val) * noise_val
                
                pred_val = net(x_t_val, t_cont_val, return_embedding=False)
                val_loss = torch.mean((pred_val - noise_val)**2)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / val_iters

        # --- C. Logging & Updates ---
        current_lr = opt.param_groups[0]['lr']
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_train_loss:.5f} (MSE:{avg_mse:.4f} NCE:{avg_nce:.4f}) | Val: {avg_val_loss:.5f} | LR: {current_lr:.2e} | Best Val: {best_val_loss:.5f}")
            # best = False
        scheduler.step(avg_val_loss)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch + 1
            # best = True
            # if epoch % 10 == 0 and best:
            #     print("  <-- Best")
        else:
            early_stop_counter += 1
            # print("")
            if early_stop_counter >= patience and scheduler._last_lr[0] < 1e-6:
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