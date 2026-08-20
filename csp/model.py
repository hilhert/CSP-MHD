import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class ModSiLU(nn.Module):
    """
    复数 ModSiLU：对模长做 SiLU，保持相位不变
    SiLU(x) = x * sigmoid(x)
    """
    def forward(self, h_real, h_imag):
        magnitude = torch.sqrt(h_real**2 + h_imag**2 + 1e-8)
        scale = F.silu(magnitude) / (magnitude + 1e-8)
        return h_real * scale, h_imag * scale

class LinearTreeAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, p=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.p = p

        # 核矩阵 B: [num_heads, head_dim, head_dim]
        # 如果 hidden_dim 不能被 num_heads 整除，head_dim 会被截断
        # 这相当于丢弃了多余的维度，但保留主要结构
        self.B = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.01)
        self.B_FUSE = nn.Parameter(torch.randn(num_heads,num_heads) * 0.01)
        self.gate = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        # ★ 因果mask：只保留下三角（包括对角线）
        #self.register_buffer('causal_mask', None)
        

    def forward(self, x):
        """
        x: [B, T, 2, H]  0: real, 1: imag
        """
        B, T, _, H = x.shape
        d = self.head_dim
        #if self.causal_mask is None or self.causal_mask.size(0) != T:
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).detach()  # [T, T]
        causal_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        

        # 1. 只取实部计算注意力，重塑为多头
        z = x[:, :, 0, :]  # [B, T, H]
        z = z.view(B, T, self.num_heads, d)  # [B, T, num_heads, d]

        # 2. 计算多头注意力分数
        zB = torch.einsum('bthd, hdd -> bthd', z, self.B)  # [B, T, num_heads, d]
        attn_scores = torch.einsum('bthd, bHhd -> bthH', zB, z)  # [B, T, num_heads, T]
        attn_scores = torch.sigmoid(attn_scores)
        
        
        # 3. 提取次对角线（用于树结构惩罚）
        subdiag = torch.diagonal(attn_scores, offset=1, dim1=1, dim2=3)  # [B, num_heads, T-1] 
        subdiag = 1-subdiag
        assert not torch.any(subdiag < 0), f"subdiag has unsuitable values! "
        assert not torch.any(subdiag > 1), f"subdiag has unsuitable values! "
        # 4. 树结构惩罚（累积和版本）
        accu = torch.cumsum(torch.exp(subdiag), dim=-1)  # [B, num_heads, T-1]
        accu = torch.cat([torch.zeros(B, self.num_heads, 1, device=x.device), accu], dim=-1)  # [B, num_heads, T]
        #print(accu[0,0,:])
        C = accu.unsqueeze(-1) - accu.unsqueeze(-2)  # [B, num_heads, T, T]# [B, num_heads, T, T]
        C = C.masked_fill(causal_mask == 0, 0)
        assert not torch.any(C < 0), f"C has negative values! min: {C.min().item()}"
        C = torch.log(1+C+1e-6)
        assert not torch.any(C < 0), f"C has negative values after log! min: {C.min().item()}"
        # 5. 应用惩罚
        attn_scores = attn_scores.permute(0, 2, 1, 3)  # [B, num_heads, T, T]
        attn_scores = 1 + F.relu(attn_scores - self.p * C)

        # 6. 跨头融合
        sim_maxsubdiag = C[:,:,-1,0] # [B, num_heads]
        zBF = torch.einsum('bh, hh -> bh', sim_maxsubdiag, self.B_FUSE)  # [B, num_heads]
        cross_attn = torch.einsum('bh, bH -> bhH', zBF, sim_maxsubdiag)  # [B, num_heads, num_heads]
        cross_attn = torch.sigmoid(cross_attn)
        cross_attn = cross_attn / (torch.sum(cross_attn, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, num_heads]

        # 更新 attn_scores
        attn_scores = torch.einsum('bhtT, bhh -> bhtT', attn_scores, cross_attn)  # [B, num_heads, T, T]

        # 因果mask + 归一化
        attn_scores = attn_scores.masked_fill(causal_mask == 0, 0)
        attn_scores = attn_scores / (torch.sum(attn_scores, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, T, T]

        # 取平均得到最终注意力
        attn_final = attn_scores.mean(dim=1)  # [B, T, T]

        # 7. 更新 x
        x_flat = x.view(B, T, -1)
        x_weighted = torch.matmul(attn_final, x_flat).view(B, T, 2, H)

        x_out = x * torch.sigmoid(self.gate) + F.silu(x_weighted)
        magnitude = torch.sqrt(x_out[:, :, 0, :]**2 + x_out[:, :, 1, :]**2 + 1e-8)
        x_out = x_out / magnitude.unsqueeze(2)

        return x_out
        
        
        
    
    

class ComplexPRLayer_ATTENTION(nn.Module):
    def __init__(self, hidden_dim, z=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z = z

        # B  kernel matrix composed up with k_proj*v_proj
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        # 门控
        self.delta_proj = nn.Linear(hidden_dim, 1)
        self.gamma_proj = nn.Linear(hidden_dim, 1)
        self.skip_gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)

    def tree_penalty(self, h_real):
        B, T, H = h_real.shape

        h_left = h_real[:, :-1, :]
        h_right = h_real[:, 1:, :]

        Bh_right = torch.matmul(h_right, self.B)
        assoc = torch.sum(h_left * Bh_right, dim=-1)
        # ★ 反转关联度：关联度越小，惩罚越大
        assoc = torch.sigmoid(-assoc)  # 压缩到 (0,1)，反序


        C = torch.zeros(B, T, T, device=h_real.device)
        for i in range(T):
            # ★ 用 max 而不是 min：因为我们要找区间内的最大断层
            C[:,i,i] = 0
            max_val = torch.full((B,), -1e6, device=h_real.device)
            for j in range(i+1, T):
                max_val = torch.max(max_val, assoc[:, j-1])
                C[:, j, i] = max_val

        # ★ 不归一化，直接返回 C
        return C

    def forward(self, h_real_seq, h_imag_seq):
        B, T, H = h_real_seq.shape

        #B_pos = torch.matmul(self.B, self.B.T)

        h_real_input = h_real_seq
        h_imag_input = h_imag_seq

        attn_scores = torch.matmul( torch.matmul(h_real_seq, self.B),  # [B, T, H]
                                    h_real_seq.transpose(1, 2)        # [B, H, T]
                                                                    )  # [B, T, T]
        attn_scores = torch.sigmoid(attn_scores)
        # ★ 应用树结构惩罚项 C (上三角矩阵)
        C = self.tree_penalty(h_real_seq)  # [B, T, T] 只对 i < j 有值

        # ★ 应用惩罚：只对 i < j 的位置（C 有值的位置）加惩罚，mask掉对角线以下。

        attn_scores = attn_scores - self.z * C
        attn_scores = 1+F.relu(attn_scores)
        mask = torch.tril(torch.ones(T, T, device=h_real_seq.device), diagonal=0).detach()  # [T, T]
        mask = mask.unsqueeze(0)  # [1, T, T]
        attn_scores = attn_scores.masked_fill(mask == 0, 0)
        
        attn_weights = attn_scores / (attn_scores.sum(dim=1, keepdim=True) + 1e-6) # [B, T, T]

        # ★ 加权和：attn_weights @ h_real_seq
        # 对于每个目标 j，用所有源 i 的权重加权
        h_real_recur = torch.matmul(attn_weights, h_real_seq)  # [B, T, H]
        h_imag_recur = torch.matmul(attn_weights, h_imag_seq)  # [B, T, H]
        '''
        # ★ 然后对每个时间步 j 做状态更新
        outputs_real, outputs_imag = [], []
        h_real_prev = torch.zeros(B, H, device=h_real_seq.device)
        h_imag_prev = torch.zeros(B, H, device=h_real_seq.device)

        for j in range(T):
            h_real_j = h_real_seq[:, j, :]
            h_imag_j = h_imag_seq[:, j, :]

            delta_t = F.softplus(self.delta_proj(h_real_j))
            alpha = torch.exp(-delta_t)
            gamma_t = torch.sigmoid(self.gamma_proj(h_real_j))

            h_real_new = alpha * h_real_prev + gamma_t * attn_weighted_real[:, j, :]
            h_imag_new = alpha * h_imag_prev + gamma_t * attn_weighted_imag[:, j, :]

            outputs_real.append(h_real_new)
            outputs_imag.append(h_imag_new)

            h_real_prev = h_real_new
            h_imag_prev = h_imag_new
       

        h_real_recur = torch.stack(outputs_real, dim=1)
        h_imag_recur = torch.stack(outputs_imag, dim=1)
        '''
        gate = torch.sigmoid(self.skip_gate).unsqueeze(0).unsqueeze(0)
        h_real_skip = h_real_input * gate + F.silu(h_real_recur) 
        h_imag_skip = h_imag_input * gate + F.silu(h_imag_recur)

        magnitude = torch.sqrt(h_real_skip**2 + h_imag_skip**2 + 1e-8)
        h_real_out = h_real_skip / magnitude
        h_imag_out = h_imag_skip / magnitude

        return h_real_out, h_imag_out
    
    
    

class ComplexPRLayer_strict(nn.Module):
    """
    复数状态传播层（ComplexPRLayer_strict）
    使用复数矩阵 W = W_real + i W_imag 来混合实部和虚部
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ★ 复数权重矩阵 W = W_real + i W_imag
        self.W_real = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.W_imag = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        # ★ 门控：输入为 2 * hidden_dim（实部 + 虚部合并）
        self.delta_proj = nn.Linear(2 * hidden_dim, 1)
        self.gamma_proj = nn.Linear(2 * hidden_dim, 1)
        self.skip_gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # 复数激活函数
        self.mod_silu = ModSiLU()

        # 初始化
        #nn.init.zeros_(self.delta_proj.weight)
        #nn.init.zeros_(self.delta_proj.bias)
        #nn.init.zeros_(self.gamma_proj.weight)
        #nn.init.zeros_(self.gamma_proj.bias)

    def forward(self, h_real_seq, h_imag_seq):
        """
        Args:
            h_real_seq, h_imag_seq: [B, T, H]
        Returns:
            h_real_out, h_imag_out: [B, T, H]
        """
        B, T, H = h_real_seq.shape

        # 保存输入用于 Skip 连接
        h_real_input = h_real_seq
        h_imag_input = h_imag_seq

        # 初始化上一时刻状态
        h_real_prev = torch.zeros(B, H, device=h_real_seq.device)
        h_imag_prev = torch.zeros(B, H, device=h_real_seq.device)

        outputs_real, outputs_imag = [], []

        for t in range(T):
            h_real_t = h_real_seq[:, t, :]
            h_imag_t = h_imag_seq[:, t, :]

            # ★ 合并实部和虚部作为门控的输入
            h_combined = torch.cat([h_real_t, h_imag_t], dim=-1)  # [B, 2*H]

            # ★ 复数矩阵乘法：W * h_prev
            W_h_real = torch.matmul(h_real_prev, self.W_real.T) - torch.matmul(h_imag_prev, self.W_imag.T)
            W_h_imag = torch.matmul(h_real_prev, self.W_imag.T) + torch.matmul(h_imag_prev, self.W_real.T)

            # ★ 门控：由合并后的向量生成
            delta_t = F.softplus(self.delta_proj(h_combined))
            alpha = torch.exp(-delta_t)
            
            # ★ gamma_t 改为周期性门控 (1 + sin(x)) / 2
            gamma_raw = self.gamma_proj(h_combined)  # [B, 1]
            gamma_t = (1 + torch.sin(gamma_raw)) / 2  # 范围 [0, 1]

            # ★ 状态更新
            h_real_new = alpha * h_real_prev + gamma_t * W_h_real
            h_imag_new = alpha * h_imag_prev + gamma_t * W_h_imag

            outputs_real.append(h_real_new)
            outputs_imag.append(h_imag_new)

            # 更新上一时刻状态
            h_real_prev = h_real_new
            h_imag_prev = h_imag_new

        # 堆叠输出
        h_real_recur = torch.stack(outputs_real, dim=1)
        h_imag_recur = torch.stack(outputs_imag, dim=1)

        # ★ 复数 ModSiLU 激活
        h_real_act, h_imag_act = self.mod_silu(h_real_recur, h_imag_recur)

        # ★ Skip 连接：gate * input + (1 - gate) * activated
        gate = torch.sigmoid(self.skip_gate).unsqueeze(0).unsqueeze(0)  # [1, 1, H]
        h_real_skip = h_real_input * gate + h_real_act 
        h_imag_skip = h_imag_input * gate + h_imag_act 

        # ★ 复数归一化
        magnitude = torch.sqrt(h_real_skip**2 + h_imag_skip**2 + 1e-8)
        h_real_out = h_real_skip / magnitude
        h_imag_out = h_imag_skip / magnitude

        return h_real_out, h_imag_out



class ComplexPRLayer(nn.Module):
    """
    Complex Propagator with Rotation block.

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
        self.theta_proj = nn.Linear(2*hidden_dim, 1)

        # Recurrence: input projection B (shared for real and imag)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decay factor (alpha = exp(log_alpha), initialized to 1)
        self.delta_proj =  nn.Linear(2*hidden_dim, 1)
        self.gamma_proj = nn.Linear(2*hidden_dim, 1)

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
        #h_real_prev = h_real_input[:,-1,:]
        #h_imag_prev = h_imag_input[:,-1,:]
        
        outputs_real, outputs_imag = [], []

        for t in range(T):
            h_real_t = h_real_seq[:, t, :]
            h_imag_t = h_imag_seq[:, t, :]

            # 1. Rotate: theta_t from input
            decision_linear_prev = torch.cat([h_real_prev,h_imag_prev],dim=-1)
            decision_linear_current = torch.cat([h_real_t,h_imag_t],dim=-1)
            theta_t = torch.tanh(self.theta_proj(decision_linear_current)) * math.pi
            cos_t, sin_t = torch.cos(theta_t), torch.sin(theta_t)

            h_real_rot = cos_t * h_real_t - sin_t * h_imag_t
            h_imag_rot = sin_t * h_real_t + cos_t * h_imag_t

            # 2. Recur: alpha * accume_state + gamma * input_projection_after_rotation  
            delta_t = F.softplus(self.delta_proj(decision_linear_prev+decision_linear_current))
            alpha   = torch.exp(-delta_t) 
              
            gamma_t = (1 + torch.sin(self.gamma_proj(decision_linear_current)))/2
            
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
    def __init__(self, vocab_size, head_dim=16, n_head=4, num_layers=3, embed_dim=32,sos_idx=-4,model_mode='atten_mhead'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = head_dim*n_head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.phase_proj   = nn.Linear(2*self.hidden_dim,self.hidden_dim)
        #self.decoder_proj = nn.Linear(hidden_dim, embed_dim)
        if model_mode == 'strict':
            self.layers = nn.ModuleList([ComplexPRLayer_strict(hidden_dim) for _ in range(num_layers)])
            print("Strict mode received! I'm createing hidden layer with modsilu and complex number matrix mul!")
        elif model_mode == 'atten':
            self.layers = nn.ModuleList([ComplexPRLayer_ATTENTION(hidden_dim) for _ in range(num_layers)])
            print("Atten mode received! I'm createing attention style hidden layer!")
        elif model_mode == 'atten_mhead':
            self.layers = nn.ModuleList([LinearTreeAttnLayer(n_head, self.hidden_dim, p=0.1) for _ in range(num_layers)])
            print("Linear Tree Atten mode received! I'm createing tree attention style hidden layer!")
        else:
            self.layers = nn.ModuleList([ComplexPRLayer(hidden_dim) for _ in range(num_layers)])
            print("No known mode explicted! Draw back to relax mode with shared weights and per domain activate!")
        #self.gate_proj = nn.Linear(2*hidden_dim,hidden_dim)
        #self.combine_proj = nn.Linear(embed_dim,hidden_dim)
        self.output_proj  = lambda x: torch.matmul(x, self.embedding.weight.T)
        #self.output_proj = nn.Linear(2*embed_dim,vocab_size)
        #self.embedding_proj = nn.Linear(hidden_dim,embed_dim)
        self.max_len = 30
        self.sos_idx=sos_idx
        self.decoder_proj = nn.Linear(self.hidden_dim,embed_dim)
        
    def fixed_rotation(self, h_real, h_imag, max_angle=20):
        """
        对输入状态做固定旋转（所有时间步一次性计算）
        h_real, h_imag: [B, T, H]
        max_angle: 最大旋转角度（度）
        """
        B, T, H = h_real.shape
        # 生成所有时间步的角度
        theta = torch.arange(T, device=h_real.device) / (T + 1) * max_angle * torch.pi / 180
        # 扩展维度以便广播
        theta = theta.view(1, T, 1)  # [1, T, 1]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        h_real_rot = cos_t * h_real - sin_t * h_imag
        h_imag_rot = sin_t * h_real + cos_t * h_imag
        return h_real_rot, h_imag_rot

    def forward(self, input_ids, target_ids=None):
        B, T_in = input_ids.shape
        x = self.embedding(input_ids)
        h = torch.tanh(self.encoder_proj(x))  #may be used in teacher forcing!
        h_real, h_imag = h, torch.zeros_like(h)
        h_real, h_imag = self.fixed_rotation(h_real,h_imag)
        x = torch.stack([h_real,h_imag],dim=2)
        

        for layer in self.layers:
            x = layer(x)

        h_real_last = x[:, -1, 0,:]
        h_imag_last = x[:, -1, 1,:]
        
        # ★ 提取相位，作为解码初始状态
        phase = torch.atan2(h_imag_last, h_real_last + 1e-8)
        h_cos, h_sin = torch.cos(phase), torch.sin(phase)
        h_phasor = torch.cat([h_cos, h_sin], dim=-1)
        hidden_out = torch.tanh(self.phase_proj(h_phasor))  # [B, H]

        if target_ids is not None:
            T_out = target_ids.shape[1]
            out_embeds = self.embedding(target_ids)
            outputs = []
            current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)  # <SOS>
            emb = self.embedding(current_token)  # [B, E]
            for t in range(T_out):
                emb_hidden = torch.tanh(self.encoder_proj(emb))  # [B, H]
                hidden_out = hidden_out + emb_hidden  # 
                emb_out = torch.tanh(self.decoder_proj(hidden_out))
                logits = self.output_proj(emb_out)
                outputs.append(logits)
                emb = out_embeds[:, t, :]  # [B, E]
            return torch.stack(outputs, dim=1)
        
        else:
            outputs = []
            current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)
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
        
        '''
        # 只取最后一个时间步的相位投影
        phase = torch.atan2(h_imag[:, -1, :], h_real[:, -1, :] + 1e-8)  # [B, H]
        h_cos, h_sin = torch.cos(phase), torch.sin(phase)
        h_phasor = torch.cat([h_cos, h_sin], dim=-1)
        hidden_state =self.phase_proj(h_phasor)  # [B, hidden_dim]
        #emb_out_norm = F.normalize(emb_out, p=2, dim=-1)  # [B, embed_dim]
        embedding_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
        if target_ids is not None:
            # 训练阶段：教师强制
            T_out = target_ids.shape[1]
            out_embeds = self.embedding(target_ids)  # [B, T_out, E]
            outputs = []
            current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)  # <SOS>
            current_emb = self.embedding(current_token)  # [B, E]
            for t in range(0,T_out):
                # 上一步的输出（教师强制）
                emb2hidden = self.combine_proj(current_emb)
                hidden_state = emb2hidden + hidden_state  # [B, embed_dim + E]   
                combined = F.normalize(self.embedding_proj(hidden_state),p=2,dim=-1)
                logits = torch.matmul(combined,embedding_norm.T)  # [B, vocab_size]
                outputs.append(logits)
                #target_emb = out_embeds[:, t, :]  # [B, E]
                # 更新隐藏状态（可选）
                current_emb  = out_embeds[:,t,:]
            return torch.stack(outputs, dim=1)

        else:
            # 推理阶段：自回归生成
            outputs = []
            current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)  # <SOS>
            current_emb = self.embedding(current_token)  # [B, E]
            for i in range(1,30): 
                emb2hidden = self.combine_proj(current_emb)
                hidden_state = emb2hidden + hidden_state  # [B, embed_dim + E]   
                combined = F.normalize(self.embedding_proj(hidden_state),p=2,dim=-1)
                logits = torch.matmul(combined,embedding_norm.T)  # [B, vocab_size]
                next_token = torch.argmax(logits, dim=-1)
                outputs.append(next_token)
                current_token = next_token
                current_emb  = self.embedding(current_token)
                #print(f"Step {i}: token {next_token.item()}")  # ★ 看有没有打印
            return torch.stack(outputs, dim=1)
        
        
  
        # ★ 提取相位，作为解码初始状态
        phase = torch.atan2(h_imag, h_real + 1e-8)
        h_cos, h_sin = torch.cos(phase), torch.sin(phase)
        h_phasor = torch.cat([h_cos, h_sin], dim=-1)
        emb_out = torch.tanh(self.phase_proj(h_phasor))  # [B,T,embed_dim]
        #emb_hidden = torch.tanh(self.encoder_proj(emb)) 
        #hidden_final = hidden_out_before[:,-1,:]
        #gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_out_before, hidden_final.unsqueeze(1).expand(-1, hidden_out_before.size(1), -1)], dim=-1)))
        #hidden_combined = gate * hidden_out_before + (1 - gate) *   hidden_final.unsqueeze(1).expand(-1, hidden_out_before.size(1), -1)
        #hidden_out = torch.tanh(self.combine_proj(hidden_combined))
        #emb_out = torch.tanh(self.decoder_proj(hidden_out))  # [B, T, embed_dim]
        emb_out_norm = F.normalize(emb_out, p=2, dim=-1)  # [B, T, embed_dim]
        embedding_norm = F.normalize(self.embedding.weight, p=2, dim=-1)  # [vocab_size, embed_dim]

        # 余弦相似度作为 logits（范围 [-1, 1]）
        logits = torch.matmul(emb_out_norm, embedding_norm.T)  # [B, T, vocab_size]
        if target_ids is not None:
            # 训练阶段：直接返回 logits
            return logits
        else:
            # 推理阶段：从 logits 中取 argmax
            outputs = torch.argmax(logits, dim=-1)  # [B, T]
            return outputs
        
        
        if target_ids is not None:
            #T_out = target_ids.shape[1]
            #out_embeds = self.embedding(target_ids)
            emb_out = torch.tanh(self.decoder_proj(hidden_out))  # [B, T, embed_dim]
            logits = self.output_proj(emb_out)  # [B, T, vocab_size]
            return logits

        else:
            outputs = []
            #current_token = torch.full((B,), self.vocab_size - 3, device=input_ids.device, dtype=torch.long)  # <SOS> 这真的没有必要哈！

            # 用编码器的最终状态作为初始状态
            #emb_hidden = hidden_out_before[:,0,:]  # [B, H]

            for t in range(T_in):
                # 1. 用上一步的 token 更新状态
                emb_hidden = hidden_out_before[:,t,:]
                
                # 2. 融合：上一步状态 + 当前时间步编码状态 + 输入嵌入
                #hidden_combined = hidden_prev + hidden_out[:, t, :] + emb_hidden
                # 写一个 t步向量版 emb_hidden 与 hidden_final的融合 
                gate = gate[:,t]
                hidden_combined = gate*emb_hidden +(1-gate)*hidden_final 
                
                
                hidden_prev = torch.tanh(self.combine_proj(hidden_combined))  # [B, H]

                # 3. 生成输出
                emb_out = torch.tanh(self.decoder_proj(hidden_prev))
                logits = self.output_proj(emb_out)
                next_token = torch.argmax(logits, dim=-1)
                outputs.append(next_token)
                current_token = next_token

            return torch.stack(outputs, dim=1)        
        
        
        
        else:
            outputs = []
    # 使用 <SOS> 作为初始 token
            current_token = torch.full((B,), self.vocab_size - 3, device=input_ids.device, dtype=torch.long)  # <SOS>
            # 初始隐藏状态从编码器取
            

            for t in range(T_in):  # 或者用 max_len 限制
                hidden_dec = hidden_out[:, t, :]  # [B, H]
                #emb = self.embedding(current_token)  # [B, E]
                #emb_hidden = torch.tanh(self.encoder_proj(emb))  # [B, H]
                #hidden_dec = hidden_dec + emb_hidden  # 累加当前输入
                emb_out = torch.tanh(self.decoder_proj(hidden_dec))
                logits = self.output_proj(emb_out)
                next_token = torch.argmax(logits, dim=-1)
                outputs.append(next_token)
               
                if next_token == self.vocab_size - 2:  # <EOS>
                    break
               
                current_token = next_token

            return torch.stack(outputs, dim=1)
'''
    
    

    
    
