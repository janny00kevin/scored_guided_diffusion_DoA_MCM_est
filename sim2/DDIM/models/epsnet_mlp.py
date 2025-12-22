import torch
import torch.nn as nn

class EpsNetMLP(nn.Module):
    def __init__(self, dim, hidden=512, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, x, t_cont):
        # x: (B, dim)
        if t_cont.dim() == 0 or t_cont.shape[-1] == 1:
            t_in = (t_cont.float().unsqueeze(-1) / 1.0)
        else:
            t_in = (t_cont.float().unsqueeze(-1))
        # t_in = t_in / 1000.0  # normalize time input
        te = self.time_emb(t_in)
        inp = torch.cat([x, te], dim=-1)
        return self.net(inp)