import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexMambaLayer(nn.Module):
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
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

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

            h_real_rot = cos_t * h_real_prev - sin_t * h_imag_prev
            h_imag_rot = sin_t * h_real_prev + cos_t * h_imag_prev

            # 2. Recur: alpha * rotated_state + gamma * input_projection
            alpha = torch.exp(self.log_alpha)
            B_real_t = self.B_proj(h_real_t)
            B_imag_t = self.B_proj(h_imag_t)

            h_real_new = alpha * h_real_rot + B_real_t
            h_imag_new = alpha * h_imag_rot + B_imag_t

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
    """Complete Complex State Propagator model."""
    def __init__(self, hidden_dim=64, output_dim=2, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([
            ComplexMambaLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(hidden_dim, output_dim)

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
            h_real, h_imag = layer(h_real, h_imag)

        # Phase decoding: read out phase information
        phase = torch.atan2(h_imag, h_real + 1e-8)
        phase_last = phase[:, -1, :]
        return self.decoder(phase_last)