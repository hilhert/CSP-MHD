import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    def __init__(self, vocab_size, hidden_dim=64, num_layers=3, embed_dim=32,sos_idx=-4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_proj = nn.Linear(embed_dim, hidden_dim)
        self.phase_proj   = nn.Linear(2*hidden_dim,hidden_dim)
        #self.decoder_proj = nn.Linear(hidden_dim, embed_dim)
        self.layers = nn.ModuleList([ComplexPRLayer(hidden_dim) for _ in range(num_layers)])
        #self.gate_proj = nn.Linear(2*hidden_dim,hidden_dim)
        #self.combine_proj = nn.Linear(embed_dim,hidden_dim)
        self.output_proj  = lambda x: torch.matmul(x, self.embedding.weight.T)
        #self.output_proj = nn.Linear(2*embed_dim,vocab_size)
        #self.embedding_proj = nn.Linear(hidden_dim,embed_dim)
        self.max_len = 30
        self.sos_idx=sos_idx
        self.decoder_proj = nn.Linear(hidden_dim,embed_dim)

    def forward(self, input_ids, target_ids=None):
        B, T_in = input_ids.shape
        x = self.embedding(input_ids)
        h = torch.tanh(self.encoder_proj(x))  #may be used in teacher forcing!
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
    
    

    
    
