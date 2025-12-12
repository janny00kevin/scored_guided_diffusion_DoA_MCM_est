import torch
import torch.nn as nn

class EpsNetUNet1D(nn.Module):
    def __init__(self, dim, time_emb_dim=128, base_ch=32):
        super().__init__()
        self.time_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.enc1 = nn.Conv1d(1, base_ch, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(base_ch, base_ch*2, kernel_size=3, padding=1)
        self.mid = nn.Conv1d(base_ch*2, base_ch*2, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(base_ch*2 + time_emb_dim, base_ch*2, kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(base_ch*2 + base_ch, base_ch, kernel_size=3, padding=1)
        self.out = nn.Conv1d(base_ch + 1, 1, kernel_size=3, padding=1)
        self.act = nn.ReLU()
    def forward(self, x, t_cont):
        # x: (B, dim)
        B = x.shape[0]
        if t_cont.dim() == 0:
            t_in = (t_cont.unsqueeze(0).float() / 1.0)
        else:
            t_in = (t_cont.float().unsqueeze(-1) / 1.0)
        te = self.time_emb(t_in)
        u = x.unsqueeze(1)  # (B,1,L)
        e1 = self.act(self.enc1(u))
        e2 = self.act(self.enc2(e1))
        m = self.act(self.mid(e2))
        te_b = te.unsqueeze(-1).expand(-1, -1, x.shape[1])
        d2_in = torch.cat([m, te_b], dim=1)
        d2 = self.act(self.dec2(d2_in))
        d1_in = torch.cat([d2, e1], dim=1)
        d1 = self.act(self.dec1(d1_in))
        out_in = torch.cat([d1, te_b[:, :1, :]], dim=1)
        out = self.out(out_in).squeeze(1)
        return out