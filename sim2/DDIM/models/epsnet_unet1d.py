import torch
import torch.nn as nn

class EpsNetUNet1D(nn.Module):
    def __init__(self, dim, time_emb_dim=128, base_ch=64):
        super().__init__()
        
        # Time Embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Helper: Conv -> GroupNorm -> SiLU
        # Added 'dilation' argument
        def conv_block(in_c, out_c, dilation=1):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, out_c),
                nn.SiLU()
            )

        # Encoder
        self.enc1 = conv_block(2, base_ch, dilation=1)           # RF: 3
        self.enc2 = conv_block(base_ch, base_ch*2, dilation=2)   # RF: 3 + 4 = 7
        
        # Middle (Wide vision)
        self.mid = conv_block(base_ch*2, base_ch*2, dilation=4)  # RF: 7 + 8 = 15

        # Decoder 
        # Dec2 uses dilation 2 to bridge the gap back
        self.dec2 = conv_block(base_ch*2 + time_emb_dim, base_ch*2, dilation=2) # RF: 15+4=19 (>16!)
        
        # Dec1 uses standard convolution for local details
        self.dec1 = conv_block(base_ch*2 + base_ch, base_ch, dilation=1)        # RF: 21

        # Final prediction
        self.out = nn.Conv1d(base_ch, 2, kernel_size=3, padding=1)  # RF: 23

    def forward(self, x, t_cont):
        # x: (Batch, 2, N)
        
        if t_cont.dim() == 0:
            t_in = t_cont.view(1, 1)
        else:
            t_in = t_cont.view(-1, 1)
            
        # Normalize time (Change 1000.0 to 50.0 if T=50)
        t_in = t_in / 1000.0  
        te = self.time_emb(t_in) # (B, 128)

        # Encoder
        e1 = self.enc1(x)   
        e2 = self.enc2(e1)  
        
        # Middle
        m = self.mid(e2)    

        # Decoder
        # Expand time embedding
        te_b = te.unsqueeze(-1).expand(-1, -1, x.shape[2])
        
        # Concat Middle + Time
        d2_in = torch.cat([m, te_b], dim=1) 
        d2 = self.dec2(d2_in) 
        
        # Concat Dec2 + Enc1 (Skip Connection)
        d1_in = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d1_in) 

        # Output
        out = self.out(d1)    
        return out