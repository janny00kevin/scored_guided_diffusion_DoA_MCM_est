import os
import torch

# Load a trained epsilon net model from saved weights
def load_trained_model(script_dir, device, N, model_type='mlp', file_name= None):

    weights_dir = os.path.join(script_dir, "weights")
    file_path = os.path.join(weights_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] Checkpoint file not found: {file_path}")

    # FIle loading
    checkpoint = torch.load(file_path, map_location=device)

    if model_type == 'unet1d':
        from models.epsnet_unet1d import EpsNetUNet1D as Net
        eps_net = Net(dim=2*N).to(device)
    else:
        from models.epsnet_mlp import EpsNetMLP as Net
        eps_net = Net(dim=2*N, hidden=1024, time_emb_dim=128).to(device)

    eps_net.load_state_dict(checkpoint['model_state_dict']); 
    eps_net.eval()

    return eps_net