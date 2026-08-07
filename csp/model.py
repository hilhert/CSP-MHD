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

            # 2. Recur: alpha * rotated_state + gamma * input_projection  fix gamma to 1 not learn_able for demostration 
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
    
class CSP_EXTEND(nn.Module):
    """Complete Complex State Propagator model (Full-Channel Cryptographic Edition)."""
    def __init__(self, vocab_size=2, hidden_dim=10, num_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 1. 可学习的词库 Token Embedding (复平面 S1 单位圆上的相位基底)
        init_angles = torch.linspace(0, 2 * torch.pi, steps=vocab_size + 1)[:-1]
        init_embeds = torch.stack([torch.cos(init_angles), torch.sin(init_angles)], dim=-1)
        self.token_embeds = nn.Parameter(init_embeds)  # [vocab_size, 2]

        # 2. 升维投影层：将 2D 复数 Token 向量扩展到 hidden_dim 个相位通道
        self.in_proj_real = nn.Linear(2, hidden_dim)
        self.in_proj_imag = nn.Linear(2, hidden_dim)

        # 3. 核心 Complex Mamba 演化主干
        self.layers = nn.ModuleList([
            ComplexPRLayer(hidden_dim) for _ in range(num_layers)
        ])

        # 4. 温度放缩因子 (替代显式 Linear 权重)
        self.logit_scale = nn.Parameter(torch.tensor(5.0))

    def get_token_vectors(self, x_ids):
        """查表提取 Token 的 2D 归一化复数向量"""
        normed_embeds = F.normalize(self.token_embeds, p=2, dim=-1) # [vocab_size, 2]
        return normed_embeds[x_ids]  # [B, T, 2]

    def forward(self, x):
        """
        Args:
            x: [B, T] 或 [B, T, 1] 离散 Token ID (0 或 1)
        Returns:
            logits: [B, vocab_size]
        """
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        x = x.long()

        # Step 1: 查表提取 Token 的 2D 复数相位向量 [B, T, 2]
        token_vecs = self.get_token_vectors(x)

        # Step 2: 映射到 hidden_dim 相位通道
        h_real = torch.tanh(self.in_proj_real(token_vecs))  # [B, T, H]
        h_imag = torch.tanh(self.in_proj_imag(token_vecs))  # [B, T, H]

        # Step 3: 多层 Complex Mamba 动力学演化
        for layer in self.layers:
            h_real, h_imag = layer(h_real, h_imag)

        # Step 4: 读取最后一个时刻 (T-1) 的隐藏层完整状态 [B, H]
        real_last = h_real[:, -1, :]  # [B, H]
        imag_last = h_imag[:, -1, :]  # [B, H]

        # 计算相位相干角，映射回 (cos, sin) 连续 2D 流形，不做任何 mean 压缩！
        phase = torch.atan2(imag_last, real_last)  # [B, H]

        # 每一个通道都保留自己的 (cos, sin) 特征向量 [B, H, 2]
        h_cos = torch.cos(phase)  # [B, H]
        h_sin = torch.sin(phase)  # [B, H]
        h_phasors = torch.stack([h_cos, h_sin], dim=-1)  # [B, H, 2]

        # Step 5: 全通道相干内积解码 (Multi-Channel Un-embedding)
        normed_embeds = F.normalize(self.token_embeds, p=2, dim=-1) # [vocab_size, 2]

        # 将每一个通道的 [B, H, 2] 向量直接与 [vocab_size, 2] 的词表向量做点积相干比对
        # einsum 计算: 'bhc, vc -> bhv' -> 得到每个通道对每个词表 Token 的相似度
        channel_logits = torch.einsum('bhc, vc -> bhv', h_phasors, normed_embeds)

        # 将 H 个通道的相干结果累加叠加出最终的决策 Logits [B, vocab_size]
        logits = channel_logits.sum(dim=1) * self.logit_scale

        return logits
    
    
    

    
    
