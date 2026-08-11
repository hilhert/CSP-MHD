import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexPRLayer(nn.Module):
    """
    Complex State Propagator (CSP) block.

    Components:
    - Rotate: element-wise complex rotation
    - Recur: complex-valued linear recurrence
    - Skip: gated skip connection with SiLU activation
    - Norm: element-wise complex normalization (unit circle projection)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Rotation
        self.theta_proj = nn.Linear(hidden_dim, 1)

        # Recurrence: input projection B (shared for real and imag)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decay factor (alpha = exp(log_alpha), initialized to 1)
        self.delta_proj =  nn.Linear(hidden_dim, 1)
        self.gamma_proj = nn.Linear(hidden_dim, 1)

        # Skip gate (per-dimension, initialized to 0.5)
        self.skip_gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Initialization
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.theta_proj.bias)

    def forward(self, h_real_seq, h_imag_seq):
        """
        Args:
            h_real_seq, h_imag_seq: [B, T, H]
        Returns:
            h_real_out, h_imag_out: [B, T, H]
        """
        B, T, H = h_real_seq.shape

        # Save input for skip connection
        h_real_input = h_real_seq
        h_imag_input = h_imag_seq

        h_real_prev = torch.zeros(B, H, device=h_real_seq.device)
        h_imag_prev = torch.zeros(B, H, device=h_real_seq.device)

        outputs_real, outputs_imag = [], []

        for t in range(T):
            h_real_t = h_real_seq[:, t, :]
            h_imag_t = h_imag_seq[:, t, :]

            # 1. Rotate: theta_t from input
            theta_t = torch.tanh(self.theta_proj(h_real_t)) * math.pi
            cos_t, sin_t = torch.cos(theta_t), torch.sin(theta_t)

            h_real_rot = cos_t * h_real_t - sin_t * h_imag_t
            h_imag_rot = sin_t * h_real_t + cos_t * h_imag_t

            # 2. Recur: alpha * accume_state + gamma * input_projection_after_rotation  
            delta_t = F.softplus(self.delta_proj(h_real_t+h_real_prev))
            alpha   = torch.exp(-delta_t) 
            gamma_t = (1 + torch.sin(self.gamma_proj(h_real_t)))/2
            
            B_real_t = self.B_proj(h_real_rot)
            B_imag_t = self.B_proj(h_imag_rot)

            h_real_new = alpha*h_real_prev  + gamma_t*B_real_t
            h_imag_new = alpha*h_imag_prev  + gamma_t*B_imag_t

            outputs_real.append(h_real_new)
            outputs_imag.append(h_imag_new)

            h_real_prev = h_real_new
            h_imag_prev = h_imag_new

        # 3. Stack recurrence outputs
        h_real_recur = torch.stack(outputs_real, dim=1)  # [B, T, H]
        h_imag_recur = torch.stack(outputs_imag, dim=1)

        # 4. Skip connection: gated input + SiLU(recur)
        gate = torch.sigmoid(self.skip_gate)  # [H]
        h_real_skip = h_real_input * gate + F.silu(h_real_recur)
        h_imag_skip = h_imag_input * gate + F.silu(h_imag_recur)

        # 5. Complex normalization: project onto unit circle
        magnitude = torch.sqrt(h_real_skip**2 + h_imag_skip**2 + 1e-8)
        h_real_out = h_real_skip / magnitude
        h_imag_out = h_imag_skip / magnitude

        return h_real_out, h_imag_out


class CSP(nn.Module):
    """
    Complete Complex State Propagator model.
    """
    
    def __init__(self, hidden_dim=64, output_dim=2, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(1, hidden_dim)
        
        #self.c_weight = nn.Parameter(torch.randn(hidden_dim, 2) * 0.02)
        #self.c_bias =  nn.Parameter(torch.zeros(hidden_dim))
        #self.masked_atan = MaskedAtan2(threshold_ratio=0.95)
        
        self.layers = nn.ModuleList([
            ComplexPRLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(2*hidden_dim, output_dim)
        
        

    def forward(self, x):
        """
        Args:
            x: [B, T, 1] binary sequence (0 or 1)
        Returns:
            logits: [B, output_dim]
        """
        h = torch.tanh(self.encoder(x))       # [B, T, H]
        h_real = h
        h_imag = torch.zeros_like(h)

        for layer in self.layers:
            h_real, h_imag = layer(h_real, h_imag)  #normed before layer output
        # Phase decoding: read out phase information
        phase = torch.atan2(h_imag[:,-1,:], h_real[:,-1,:]) 
        #phase = self.masked_atan(h_imag[:,-1,:], h_real[:,-1,:])
        x_dec  =  torch.cat([torch.cos(phase),torch.sin(phase)],dim=-1) 
        #x = torch.stack([h_imag[:,-1,:],h_real[:,-1,:]],dim=-1)
        
        #x = torch.einsum('bhc, hc -> bh', x, self.c_weight) + self.c_bias
        #warp_unwarp_phase_last = unwarp_phase[:, -1, :]
        #return self.decoder(warp_unwarp_phase_last)
        return self.decoder(x_dec)
    
class CSP_Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, num_layers=3, embed_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_proj = nn.Linear(embed_dim, hidden_dim)
        self.phase_proj   = nn.Linear(2*hidden_dim,hidden_dim)
        self.decoder_proj = nn.Linear(hidden_dim, embed_dim)
        self.layers = nn.ModuleList([ComplexPRLayer(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, target_ids=None):
        B, T_in = input_ids.shape
        x = self.embedding(input_ids)
        h = torch.tanh(self.encoder_proj(x))
        h_real, h_imag = h, torch.zeros_like(h)

        for layer in self.layers:
            h_real, h_imag = layer(h_real, h_imag)

        h_real_last = h_real[:, -1, :]
        h_imag_last = h_imag[:, -1, :]

        # ★ 提取相位，作为解码初始状态
        phase = torch.atan2(h_imag_last, h_real_last + 1e-8)
        h_cos, h_sin = torch.cos(phase), torch.sin(phase)
        h_phasor = torch.cat([h_cos, h_sin], dim=-1)
        hidden_out = torch.tanh(self.phase_proj(h_phasor))  # [B, H]

        if target_ids is not None:
            T_out = target_ids.shape[1]
            out_embeds = self.embedding(target_ids)
            outputs = []
            for t in range(T_out):
                emb = out_embeds[:, t, :]  # [B, E]
                emb_hidden = torch.tanh(self.encoder_proj(emb))  # [B, H]
                hidden_out = hidden_out + emb_hidden  # 
                emb_out = torch.tanh(self.decoder_proj(hidden_out))
                logits = self.output_proj(emb_out)
                outputs.append(logits)
            return torch.stack(outputs, dim=1)

        else:
            outputs = []
            sos_idx = self.vocab_size - 3
            eos_idx = self.vocab_size - 2
            current_token = torch.full((B,), sos_idx, device=input_ids.device, dtype=torch.long)
            for _ in range(20):
                emb = self.embedding(current_token)
                emb_hidden = torch.tanh(self.encoder_proj(emb))
                hidden_out = hidden_out + emb_hidden
                emb_out = torch.tanh(self.decoder_proj(hidden_out))
                logits = self.output_proj(emb_out)
                next_token = torch.argmax(logits, dim=-1)
                outputs.append(next_token)
                current_token = next_token
            return torch.stack(outputs, dim=1)

    
    

    
    
