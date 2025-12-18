import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3):
        super().__init__()
        self.act = nn.SiLU() # Swish activation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        # First Conv
        h = self.conv1(self.act(x))
        h = self.norm1(h)
        
        # Add Time Embedding (Broadcasting)
        # t_emb: (B, out_ch) -> (B, out_ch, 1)
        h += self.time_mlp(self.act(t_emb))[:, :, None]
        
        # Second Conv
        h = self.conv2(self.act(h))
        h = self.norm2(h)
        
        return h + self.shortcut(x)

class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) for MultiheadAttention
        B, C, L = x.shape
        x_in = x.permute(0, 2, 1)
        x_ln = self.ln(x_in)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_in
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)

class EpsNetResUNet1D(nn.Module):
    def __init__(self, dim=2, time_emb_dim=128, base_ch=128):
        super().__init__()
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial Conv
        self.inc = nn.Conv1d(2, base_ch, 3, padding=1) # Input channels = 2 (Real, Imag)

        # Downsample Phase (No pooling, just processing)
        self.down1 = ResidualBlock1D(base_ch, base_ch, time_emb_dim)
        self.down2 = ResidualBlock1D(base_ch, base_ch*2, time_emb_dim)
        
        # Middle Phase (Attention)
        self.mid1 = ResidualBlock1D(base_ch*2, base_ch*2, time_emb_dim)
        self.attn = SelfAttention1D(base_ch*2)
        self.mid2 = ResidualBlock1D(base_ch*2, base_ch*2, time_emb_dim)

        # Upsample Phase
        self.up1 = ResidualBlock1D(base_ch*3, base_ch, time_emb_dim)      # Input: 384 -> Output: 128
        self.up2 = ResidualBlock1D(base_ch*2, base_ch, time_emb_dim) # Concat: ch + ch -> ch*2

        # Final Output
        self.outc = nn.Conv1d(base_ch, 2, 1) # Output channels = 2

    def forward(self, x, t):
        # x: (B, 2, N)
        # t: (B) or (B, 1) - raw timestep (e.g. 0 to 1000)
        
        # Process Time
        t = t.squeeze()
        if t.dim() == 0: t = t.unsqueeze(0)
        t_emb = self.time_mlp(t)

        # Input
        x1 = self.inc(x)
        
        # Down
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        
        # Mid
        x3 = self.mid1(x3, t_emb)
        x3 = self.attn(x3)
        x3 = self.mid2(x3, t_emb)
        
        # Up (Concat with skip connections)
        # x3: (B, 256, N), x2: (B, 256, N) -> cat -> (B, 512, N)
        x = self.up1(torch.cat([x3, x2], dim=1), t_emb)
        
        # x: (B, 128, N), x1: (B, 128, N) -> cat -> (B, 256, N)
        x = self.up2(torch.cat([x, x1], dim=1), t_emb)
        
        return self.outc(x)