import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearMhRBFKAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim ,merge_attention=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.logp = nn.Parameter(torch.tensor(0.5))

        # K_proj * V_proj =  B: [num_heads, head_dim, head_dim]
       
        self.B = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.01)
        self.B_FUSE = nn.Parameter(torch.randn(num_heads,num_heads) * 0.01)
        self.gate = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.dp    = nn.Parameter(torch.tensor(-5.0))
        self.merge_attention=merge_attention

    def forward(self, x):
        """
        x: [B, T, 2, H]  0: real, 1: imag
        """
        B, T, _, H = x.shape
        d = self.head_dim
        #if self.causal_mask is None or self.causal_mask.size(0) != T:
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).detach()  # [T, T]
        causal_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        

        # 1. take both real part and imag part to build attentions，multi_head
        z , zi  =   x[:, :, 0, :], x[:, :, 1, :]  # [B, T, H]
        z   =   z.view(B, T, self.num_heads, d)  # [B, T, num_heads, d]
        zi  =   zi.view(B, T, self.num_heads, d)   
        #diag = torch.eye(d, device=x.device).unsqueeze(0)  # [1, d, d]
        Bb = torch.matmul(self.B,self.B.permute(0,2,1)) #+ torch.sigmoid(self.dp)*diag
        
        # 2. compute the attention score
        zBh   =  torch.einsum('bthd, hdd -> bthd', z, self.B)  # [B, T, num_heads, d]
        ziBh   =  torch.einsum('bthd, hdd -> bthd', zi, self.B)
        
        dist_unformulated_r = torch.einsum('bthd, bHhd -> bthH', zBh, z).permute(0,2,1,3)  # [B, num_heads,T, T]
        dist_unformulated_i = torch.einsum('bthd, bHhd -> bthH', ziBh, zi).permute(0,2,1,3)  # [B, num_heads,T, T]
        dist_unformulated  = dist_unformulated_r + dist_unformulated_i  # Regard as dual channel info!
        #attn_scores = torch.sigmoid(attn_scores)
        diag  = torch.diagonal(dist_unformulated,dim1=2,dim2=3) #[B,num_heads,T]
        
        dist = diag.unsqueeze(-1) + diag.unsqueeze(-2) - 2*dist_unformulated  # mahalonobis distance is all positive 
        
        
        
        # 3. extract sub diag （for tree penalty）
        subdiag = torch.diagonal(dist, offset=1, dim1=2, dim2=3)  # [B, num_heads, T-1] 
        #subdiag = -subdiag
        #assert not torch.any(subdiag < 0), f"subdiag has unsuitable values! "
        #assert not torch.any(subdiag > 1), f"subdiag has unsuitable values! "
        # 4. tree penaly construct using accum differ matrix (log sum exp)!
        accu = torch.cumsum(torch.exp(subdiag), dim=-1)  # [B, num_heads, T-1]
        accu = torch.cat([torch.zeros(B, self.num_heads, 1, device=x.device), accu], dim=-1)  # [B, num_heads, T]
        #print(accu[0,0,:])
        C = accu.unsqueeze(-1) - accu.unsqueeze(-2)  # [B, num_heads, T, T]# [B, num_heads, T, T]
        C = C.masked_fill(causal_mask == 0, 0)
        #assert not torch.any(C < 0), f"C has negative values! min: {C.min().item()}"
        C = torch.log(1+C+1e-6)  # sim max, log_sum_exp
        #assert not torch.any(C < 0), f"C has negative values after log! min: {C.min().item()}"
        # 5. apply penalty
        #attn_scores = torch.sigmoid(attn_scores)
        
        dist_rect = dist + torch.exp(self.logp) * C
        
        # 6. cross head attention!
        sim_maxdist_ph = C[:,:,-1,0] # [B, num_heads]  sim maxium scores in subdiag selected!
        #c_dist         = torch.einsum('bh, bH -> bhH', sim_maxdist_ph, sim_maxdist_ph)*self.B_FUSE
        
        zBF = torch.einsum('bh, hh -> bh', sim_maxdist_ph, self.B_FUSE)  # [B, num_heads]
        cross_attn = torch.einsum('bh, bH -> bhH', zBF, sim_maxdist_ph)  # [B, num_heads, num_heads]
        cross_attn = 1+F.silu(cross_attn)
        cross_attn = cross_attn / (torch.sum(cross_attn, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, num_heads]

        # update synthetic dist and generate attn acores! 
        
        dist_mixture = torch.einsum('bhtT, bhh -> bhtT', dist_rect, cross_attn)  # [B, num_heads, T, T]
        attn_scores = torch.exp(-dist_mixture)
        
        #attn_scores = torch.exp(-(torch.sin(dist_mixture)+1)/2)  # May help ... Seriously!
        
        # causual mask + normalize
        attn_scores = attn_scores.masked_fill(causal_mask == 0, 0)
        attn_scores = attn_scores / (torch.sum(attn_scores, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, T, T]
           
        # Dyanmic the attend confused attention! 
        
        z_flat = x.view(B, T, -1)
        
        if self.merge_attention: 
            attn_final = attn_scores.mean(dim=1) # [B, T, T]
            z_weighted = torch.matmul(attn_final, z_flat).view(B, T, 2, H)
        
        else:
            zB_f = torch.einsum('bThd, btTh -> bhtd', z, attn_scores.permute(0,2,3,1))
            ziB_f = torch.einsum('bThd, btTh -> bhtd', zi, attn_scores.permute(0,2,3,1))

            z_weighted =  torch.cat([zB_f.view(B,T,-1), ziB_f.view(B,T,-1)],dim=-1)

        z_out = z_flat.view(B,T,2,H) * torch.sigmoid(self.gate) + F.silu(z_weighted.view(B,T,2,H))
        magnitude = torch.sqrt(z_out[:, :, 0, :]**2 + z_out[:, :, 1, :]**2 + 1e-8)
        z_out = z_out / magnitude.unsqueeze(2)

        return z_out

    
class CSP_Hidden_FAST(nn.Module):
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

    def forward(self, x):
        """
        Args:
            x: [B, T, 2, H]  0: real, 1: imag
        Returns:
            x: [B, T, 2, H]
        """
        B, T, _, H = x.shape

        #1. Rotation with imposed statistics!
  
        theta_all = torch.tanh(self.theta_proj(torch.cumsum(x.view(B,T,-1),dim=1)/torch.arange(1,T+1,device=x.device).unsqueeze(-1).unsqueeze(0).detach())) * math.pi  # [B, T, 1]
        cos_a, sin_a = torch.cos(theta_all), torch.sin(theta_all)  # [B, T, 1]

        # construct of rotation matrix
        Rot1 = torch.cat([cos_a, -sin_a], dim=-1)  # [B, T, 1, 2]
        Rot2 = torch.cat([sin_a, cos_a], dim=-1)   # [B, T, 1, 2]
        Rot = torch.stack([Rot1, Rot2], dim=-2)    # [B, T, 2, 2]

       
        x_rot = torch.einsum('btih, btji -> btjh', x, Rot)  
        
        
        #2. Element construct after rotation
        gamma_all       =     (1 + torch.sin(self.gamma_proj(x_rot.view(B,T,-1)))).unsqueeze(-1)/2 
        x_g             =     self.B_proj(x_rot)* gamma_all
        
        
        #3. Mamba style recur
        delta_all       =     F.softplus(self.delta_proj(x_g.view(B,T,-1)))  # [B, T, 1]
        alpha_all       =     torch.exp(-delta_all).unsqueeze(-1)                # [B, T, 1] [B, T, 1]
        alpha_cumprod   =     torch.cumprod(alpha_all,dim=1) 
        alpha_cumpshift =     torch.cat([torch.ones(B,1,1,1,device=x.device).detach(),alpha_cumprod[:,:-1,:,:]],dim=1) 
        
        x_recur = alpha_cumprod*torch.cumsum(x_g / (alpha_cumpshift+1e-9), dim=1)
        
       
        
        #4. Skip connection with elementwise magnitude normalization
        gate = torch.sigmoid(self.skip_gate) 
        x_out = x * gate + F.silu(x_recur)       
        magnitude = torch.sqrt(x_out[:, :, 0, :]**2 + x_out[:, :, 1, :]**2 + 1e-8)
        x_out = x_out / magnitude.unsqueeze(-2)  

        return x_out    
    
class CSP_BLOCK(nn.Module):
    def __init__(self,block_config): 
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        for layer_c in block_config["layers"]:
            model_mode = layer_c["name"]
            hidden_dim = layer_c["hidden_dim"]
            n_head     = layer_c["num_heads"]
            
        
            if model_mode == 'CSP_Hidden_FAST':
                self.layers.append(CSP_Hidden_FAST(hidden_dim))
                print("Complex mamba with rotation mode received! I'm createing CSP style hidden layer!")
            
            elif model_mode == 'LinearMhRBFKAttnLayer':
                self.layers.append(LinearMhRBFKAttnLayer(n_head, hidden_dim))
                print("Multi head rbf Atten mode received! I'm createing attention style hidden layer with mhead!")
            
            else:
                self.layers.append(CSP_Hidden_FAST(hidden_dim))
                print("No known mode explicted! Draw back to csp hidden fast mode with shared weights and per domain activate!")

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    

class CSP_head(nn.Module): 
    def __init__(self,embedding,embed_dim,hidden_dim,extend_historical):
        super().__init__()
        self.embedding = embedding
        self.emb_K   = nn.Linear(embed_dim,embed_dim)
        self.ahidden_Q = nn.Linear(2*hidden_dim, embed_dim)
        self.emb_V   = nn.Linear(embed_dim,embed_dim)
        self.extend_historical = extend_historical
        self.tail_weight = nn.Parameter(torch.ones(1)*0.5)
         #                 Creating logics with similarity!  
        self.output_proj  = lambda x: torch.matmul(x, self.embedding.weight.T)
        
    def forward(self,x,hidden_out): 
        T_in = x.shape[1]
        frozen_embedding = self.embedding.weight.detach()  # [vocab, H]
        emb_K_vocab = self.emb_K(frozen_embedding)  # [vocab, H]
        emb_K_vocab_norm = emb_K_vocab / (emb_K_vocab.norm(dim=-1, keepdim=True) + 1e-6)
        emb_V_vocab = self.emb_V(frozen_embedding) +frozen_embedding # [vocab, H]
        #emb_V_vocab_norm = emb_V_vocab / (emb_V_vocab.norm(dim=-1, keepdim=True) + 1e-6)

        if self.extend_historical > 0 and T_in > self.extend_historical:
            hist_sup = x[:, -self.extend_historical-1:-1, :, :].detach()  # [B, hist, 2, H]
            phase_sup = torch.atan2(hist_sup[:, :, 1, :], hist_sup[:, :, 0, :] + 1e-8)  # [B, hist, H]
            sup_cos, sup_sin = torch.cos(phase_sup), torch.sin(phase_sup)
            sup_phasor = torch.cat([sup_cos, sup_sin], dim=-1)  # [B, hist, 2*H]
            emb_extend = self.ahidden_Q(sup_phasor)  # [B, hist, H]
            emb_K_tail = self.emb_K(emb_extend)  # [B, hist, H]
            emb_K_tail_norm = emb_K_tail / (emb_K_tail.norm(dim=-1, keepdim=True) + 1e-6)
            emb_V_tail = self.emb_V(emb_extend)   # [B, hist, H]
            #emb_V_tail_norm = emb_V_tail / (emb_V_tail.norm(dim=-1, keepdim=True) + 1e-6)

        else:
            emb_extend = None        

        phase = torch.atan2(hidden_out[:, 1, :], hidden_out[:, 0, :] + 1e-8)
        h_cos, h_sin = torch.cos(phase), torch.sin(phase)
        h_phasor = torch.cat([h_cos, h_sin], dim=-1)  # [B, 2*H]
        hidden_Q = torch.tanh(self.ahidden_Q(h_phasor))  # [B, H]
        # normalize
        hidden_Q_norm = hidden_Q / (hidden_Q.norm(dim=-1, keepdim=True) + 1e-6)

        # vocab distance + attention
        dist_sq_vocab = torch.cdist(hidden_Q_norm, emb_K_vocab_norm, p=2) ** 2  # [1, vocab]
        attn_vocab = torch.softmax(-dist_sq_vocab, dim=-1)  # [1, vocab]
        emb_out_vocab = torch.matmul(attn_vocab,emb_V_vocab)  # [1, H]

        # ========== hist tail ==========
        #    historical step
        if emb_extend is not None:

            # tail distanced attention
            dist_sq_tail = torch.cdist(hidden_Q_norm.unsqueeze(1), emb_K_tail_norm, p=2).squeeze(1) ** 2  # [B, hist,hit]
            attn_tail = torch.softmax(-dist_sq_tail, dim=-1)  # [B, hist]

            # ========== merge vocab and tail==========
            # method：compute vocab attention and tail attention，and merge again

            emb_out_tail = torch.matmul(attn_tail.unsqueeze(1), emb_V_tail).squeeze(1)  # [B, H]


            tail_weight = torch.sigmoid(self.tail_weight)
            #print(tail_weight.shape)
            emb_out =  emb_out_vocab + tail_weight * emb_out_tail



        else:
            emb_out = emb_out_vocab  # [1, H]


        logits = self.output_proj(emb_out)
        
        return logits 
        
    
    
    
    
class CSP_Seq2Seq(nn.Module):
    def __init__(self, model_config, vocab_config ,train_method='teacher_forcing'):
        super().__init__()
        self.train_method =  train_method
        self.vocab_size   =  vocab_config["vocab_size"]
        self.hidden_dim   =  model_config["global"]["hidden_dim"]
        self.embed_dim    =  model_config["global"]["embed_dim"]
        self.embedding    =  nn.Embedding(self.vocab_size, self.embed_dim)
        with torch.no_grad():
            self.embedding.weight[vocab_config["<SOS>"]].zero_()
            self.embedding.weight[vocab_config["<EOS>"]].requires_grad = False
        
        # shared encoder for raw tokens inputs even in tecahr forcing!
        self.encoder_proj = nn.Linear(self.embed_dim, self.hidden_dim)
       
        self.blocks   = nn.ModuleList([CSP_BLOCK(model_config["block"]) for _ in range(model_config["global"]["block_num"])])
       
        #                 Specialised Config for Teacher Forcing 
        self.max_len = model_config["global"]["max_len"]
        self.angle_step = model_config["global"]["angle_step"]
        self.sos_idx=vocab_config["<SOS>"]
        self.pad_idx=vocab_config["<PAD>"]
        self.eos_idx=vocab_config["<EOS>"]
        
        # decay factor of historical hidden output!
        self.alpha_proj = nn.Linear(self.hidden_dim,1)
        
        
        self.head =  CSP_head(self.embedding,self.embed_dim,self.hidden_dim,model_config["head"]["extend_historical"])             
        
      
        
    def rotation(self, h_real, h_imag, angle_step,pre_angle=None):
        """
        roation with all time sequence inputs 
        h_real, h_imag: [B, T, H]
        max_angle: maximum rotation angle
        """
        if pre_angle is None:
            B, T, H = h_real.shape
            # generate angles across T
            theta = torch.arange(T, device=h_real.device) *angle_step * torch.pi
            # boradcast
            theta = theta.view(1, T, 1)  # [1, T, 1]
        else:
            theta = torch.full((1, 1), (pre_angle + angle_step) * torch.pi, device=h_real.device)
            
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        h_real_rot = cos_t * h_real - sin_t * h_imag
        h_imag_rot = sin_t * h_real + cos_t * h_imag
        
        return h_real_rot, h_imag_rot


    
    def forward(self, input_ids, target_ids=None):
        B, T_in = input_ids.shape
        x = self.embedding(input_ids)
        h = torch.tanh(self.encoder_proj(x))
        h_real, h_imag = h, torch.zeros_like(h)
        h_real, h_imag = self.rotation(h_real, h_imag, angle_step=self.angle_step)
        x = torch.stack([h_real, h_imag], dim=2)

        for block in self.blocks:
            x = block(x)

    
        
        if self.train_method == 'teacher_forcing':
            # ========== vocab part ==========                
            hidden_out = x[:, -1, :, :]  # [B, 2, H]
            if target_ids is not None:
                T_out = target_ids.shape[1]
                out_embeds = self.embedding(target_ids)
                outputs = []
                current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)
                emb = self.embedding(current_token)
                out_embeds = torch.cat([emb.unsqueeze(1), out_embeds], dim=1)
                emb_hidden = torch.tanh(self.encoder_proj(out_embeds))
                emb_hr, emb_hi = self.rotation(emb_hidden, torch.zeros_like(emb_hidden), self.angle_step, None)
                emb_hidden = torch.stack([emb_hr, emb_hi], dim=-2)

             
                for t in range(T_out):
                    alpha = torch.sigmoid(self.alpha_proj(hidden_out[:, 0, :]))
                    hidden_out = torch.stack([alpha, alpha], dim=1) * hidden_out + F.silu(emb_hidden[:, t, :, :])
                    logits = self.head(x,hidden_out)
                    outputs.append(logits)
                    emb = out_embeds[:, t, :]
                    magnitude = torch.sqrt(hidden_out[:, 0, :]**2 + hidden_out[:, 1, :]**2 + 1e-8)
                    hidden_out = hidden_out / magnitude.unsqueeze(1)

                return torch.stack(outputs, dim=1)

            else:
                # ========== reasoning ==========
                outputs = []
                current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)
                pre_angle = -self.angle_step
                for t in range(self.max_len):
                    emb = self.embedding(current_token)
                    emb_hidden = torch.tanh(self.encoder_proj(emb))
                    emb_hr, emb_hi = self.rotation(emb_hidden, torch.zeros_like(emb_hidden), self.angle_step, pre_angle=pre_angle)
                    emb_hidden = torch.stack([emb_hr, emb_hi], dim=-2)
                    alpha = torch.sigmoid(self.alpha_proj(hidden_out[:, 0, :]))
                    hidden_out = torch.stack([alpha, alpha], dim=1) * hidden_out + F.silu(emb_hidden)
                    logits = self.head(x,hidden_out)
                    next_token = torch.argmax(logits, dim=-1)
                    outputs.append(next_token)
                    current_token = next_token
                    pre_angle += self.angle_step
                    magnitude = torch.sqrt(hidden_out[:,0,:]**2 + hidden_out[:,1,:]**2 + 1e-8)
                    hidden_out = hidden_out/magnitude.unsqueeze(1)

                return torch.stack(outputs, dim=1)
        else:
            # ========== non teacher forcing ==========
            emb_out_norm = F.normalize(hidden_out[:, 0, :], p=2, dim=-1)
            embedding_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
            logits = torch.matmul(emb_out_norm, embedding_norm.T)
            if target_ids is not None:
                return logits
            else:
                return torch.argmax(logits, dim=-1)    
    
