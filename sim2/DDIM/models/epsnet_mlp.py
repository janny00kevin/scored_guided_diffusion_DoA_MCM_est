import torch
import torch.nn as nn

class EpsNetMLP(nn.Module):
    def __init__(self, dim, hidden=512, time_emb_dim=128):
        super().__init__()
        # Time embedding block (unchanged)
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU()
        )
        
        # 1. Feature Extractor (The "Encoder")
        # This produces the latent vector 'z' we want to contrast
        self.feature_net = nn.Sequential(
            nn.Linear(dim + time_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        
        # 2. Output Head (The "Decoder")
        # Projects 'z' back to the noise dimension
        self.head = nn.Linear(hidden, dim)

    def forward(self, x, t_cont, return_embedding=False):
        # x: (B, dim)
        # Handle time input shape normalization (unchanged logic)
        if t_cont.dim() == 0 or (t_cont.dim() == 1 and t_cont.shape[-1] == 1):
             # Ensure shape (B, 1) if scalar or (B,) passed
             if t_cont.numel() == 1:
                 t_in = t_cont.float().view(1, 1).expand(x.shape[0], -1)
             else:
                 t_in = t_cont.float().view(-1, 1)
        else:
            t_in = (t_cont.float().unsqueeze(-1))
        t_in = t_in / 1000.0  # normalize time input
        te = self.time_emb(t_in)
        inp = torch.cat([x, te], dim=-1)
        
        # 1. Extract Features (Embedding z)
        z = self.feature_net(inp)
        # 2. Predict Noise
        out = self.head(z)
        if return_embedding:
            return out, z
        else:
            return out

class LatentEpsNet(nn.Module):
    def __init__(self, dim, hidden=512, time_emb_dim=128):
        super().__init__()
        # Time embedding block (unchanged)
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim), 
            nn.ReLU(), 
            nn.Linear(time_emb_dim, time_emb_dim), 
            nn.ReLU()
        )
        
        # --- 1. Feature Extractor (Encoder) ---
        # Expanded to be a full MLP itself (3 layers)
        # Input: Noisy State + Time -> Output: Latent Feature z (size: hidden)
        self.feature_net = nn.Sequential(
            nn.Linear(dim + time_emb_dim, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, hidden), # Added layer to match original depth
            nn.ReLU()
        )
        
        # --- 2. Output Head (Decoder) ---
        # Expanded to be a full MLP itself (3 layers)
        # Input: Latent Feature z -> Output: Predicted Noise (size: dim)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t_cont, return_embedding=False):
        # x: (B, dim)
        # Handle time input shape normalization
        if t_cont.dim() == 0 or (t_cont.dim() == 1 and t_cont.shape[-1] == 1):
             # Ensure shape (B, 1) if scalar or (B,) passed
             if t_cont.numel() == 1:
                 t_in = t_cont.float().view(1, 1).expand(x.shape[0], -1)
             else:
                 t_in = t_cont.float().view(-1, 1)
        else:
            t_in = t_cont.float().unsqueeze(-1)
            
        t_in = t_in / 1000.0  # normalize time input
        
        # Create Time Embedding
        te = self.time_emb(t_in)
        
        # Concatenate Input + Time
        inp = torch.cat([x, te], dim=-1)
        
        # 1. Extract Features (Encoder)
        z = self.feature_net(inp)
        
        # 2. Predict Noise (Decoder)
        out = self.head(z)
        
        if return_embedding:
            return out, z
        else:
            return out