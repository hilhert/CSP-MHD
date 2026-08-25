import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearMhRBFKAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, p=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.logp = nn.Parameter(torch.tensor(0.5))

        # K_proj * V_proj =  B: [num_heads, head_dim, head_dim]
       
        self.B = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.01)
        self.B_FUSE = nn.Parameter(torch.randn(num_heads,num_heads) * 0.01)
        self.gate = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        

    def forward(self, x):
        """
        x: [B, T, 2, H]  0: real, 1: imag
        """
        B, T, _, H = x.shape
        d = self.head_dim
        #if self.causal_mask is None or self.causal_mask.size(0) != T:
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).detach()  # [T, T]
        causal_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        

        # 1. take real part to build attentions，multi_head
        z , zi  =   x[:, :, 0, :], x[:, :, 1, :]  # [B, T, H]
        z   =   z.view(B, T, self.num_heads, d)  # [B, T, num_heads, d]
        zi  =   zi.view(B, T, self.num_heads, d)   
        
        # 2. compute the attention score
        zB    = torch.einsum('bthd, hdd -> bthd', z, self.B)  # [B, T, num_heads, d]
        ziB   = torch.einsum('bthd, hdd -> bthd', zi, self.B)
        
        dist_unformulated_r = torch.einsum('bthd, bHhd -> bthH', zB, z).permute(0,2,1,3)  # [B, num_heads,T, T]
        dist_unformulated_i = torch.einsum('bthd, bHhd -> bthH', ziB, zi).permute(0,2,1,3)  # [B, num_heads,T, T]
        dist_unformulated  = dist_unformulated_r + dist_unformulated_i  # Regard as dual channel info!
        #attn_scores = torch.sigmoid(attn_scores)
        diag  = torch.diagonal(dist_unformulated,dim1=2,dim2=3) #[B,num_heads,T]
        
        dist = diag.unsqueeze(-1) + diag.unsqueeze(-2) - 2*dist_unformulated  # suppose all positive 
        
        
        
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
        assert not torch.any(C < 0), f"C has negative values! min: {C.min().item()}"
        C = torch.log(1+C+1e-6)
        assert not torch.any(C < 0), f"C has negative values after log! min: {C.min().item()}"
        # 5. apply penalty
        #attn_scores = torch.sigmoid(attn_scores)
        
        dist_rect = dist + torch.exp(self.logp) * C

        # 6. cross head attention!
        sim_maxsubdiag = C[:,:,-1,0] # [B, num_heads]  sim maxium scores in subdiag selected!
        zBF = torch.einsum('bh, hh -> bh', sim_maxsubdiag, self.B_FUSE)  # [B, num_heads]
        cross_attn = torch.einsum('bh, bH -> bhH', zBF, sim_maxsubdiag)  # [B, num_heads, num_heads]
        cross_attn = 1+F.relu(cross_attn)
        cross_attn = cross_attn / (torch.sum(cross_attn, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, num_heads]

        # update synthetic dist and generate attn acores! 
        dist_mixture = torch.einsum('bhtT, bhh -> bhtT', dist_rect, cross_attn)  # [B, num_heads, T, T]
        attn_scores = torch.exp(-dist_mixture)
        
        # causual mask + normalize
        attn_scores = attn_scores.masked_fill(causal_mask == 0, 0)
        attn_scores = attn_scores / (torch.sum(attn_scores, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, T, T]
       
        

        # pick the final attn map after attention weight average
        attn_final = attn_scores[:,-1,:,:]  # [B, T, T]
       
        # 7. update x
        x_flat = x.view(B, T, -1)
        x_weighted = torch.matmul(attn_final, x_flat).view(B, T, 2, H)

        x_out = x * torch.sigmoid(self.gate) + F.silu(x_weighted)
        magnitude = torch.sqrt(x_out[:, :, 0, :]**2 + x_out[:, :, 1, :]**2 + 1e-8)
        x_out = x_out / magnitude.unsqueeze(2)

        return x_out



class LinearRBFAttnLayer(nn.Module):
    def __init__(self,  hidden_dim, p=1):
        super().__init__()
        self.logp = nn.Parameter(torch.tensor(0.5))
        self.sigma_proj_K = nn.Linear(2*hidden_dim,1)
        self.sigma_proj_V = nn.Linear(2*hidden_dim,1)
        self.gate = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        

    def forward(self, x):
        """
        x: [B, T, 2, H]  0: real, 1: imag
        """
        B, T, _, H = x.shape
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).detach()  # [T, T]
        causal_mask = mask.unsqueeze(0)  # [ 1, T, T]
        
        p = torch.exp(self.logp)
        sigma_T_K = torch.exp(self.sigma_proj_K(x.view(B,T,-1))).squeeze(-1) #[B,T]
        sigma_T_V = torch.exp(self.sigma_proj_V(x.view(B,T,-1))).squeeze(-1)
        
        #sigma_T = torch.clamp(sigma_T, min=0.1, max=5.0)
        sigma_tT = torch.einsum('bt , bT -> btT', sigma_T_K, sigma_T_V) #[B,T,T]
      
        dist = torch.cdist(x[:, :, 0, :], x[:, :, 0, :], p=2)**2 + torch.cdist(x[:, :, 1, :], x[:, :, 1, :], p=2)**2
        dist          =  dist.squeeze(-1)/(sigma_tT+1e-6)  #[B,T,T]
        dist = torch.clamp(dist, max=10.0) 
        #print(f"x shape: {x.shape}, dist shape: {dist.shape}, causal_mask shape: {causal_mask.shape}")
        
        assoc         =    torch.exp(torch.diagonal(dist, offset=1,dim1=1,dim2=2))
        
        accu          =    torch.cumsum(assoc, dim=-1)  # [B, T-1]
        accu          =    torch.cat([torch.zeros(B, 1, device=x.device), accu], dim=-1)  # [B, T]
        
        #print(f"assoc shape: {assoc.shape}, accu shape: {accu.shape}, causal_mask shape: {causal_mask.shape}")
        C = accu.unsqueeze(-1) - accu.unsqueeze(-2)  # [B, T, T]
        C = C.masked_fill(causal_mask == 0, 0)
        assert not torch.any(C < 0), f"C has negative values! min: {C.min().item()}"
        C = torch.log(1 + C + 1e-6)
   
        
        dist          =   dist + p*C
    
       
        attn_scores   =  torch.exp(-dist).masked_fill(causal_mask == 0, 0)
        
        
        attn_final = attn_scores / (torch.sum(attn_scores, dim=-1, keepdim=True) + 1e-6)  # [B, T, T]
       
        x_flat = x.view(B, T, -1)
        x_weighted = torch.matmul(attn_final, x_flat).view(B, T, 2, H)

        x_out = x * torch.sigmoid(self.gate) + F.silu(x_weighted)
        magnitude = torch.sqrt(x_out[:, :, 0, :]**2 + x_out[:, :, 1, :]**2 + 1e-8)
        x_out = x_out / magnitude.unsqueeze(2)

        return x_out


    
class LinearTreeAttnLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, p=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.p = p

        # K_proj * V_proj =  B: [num_heads, head_dim, head_dim]
       
        self.B = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.01)
        self.B_FUSE = nn.Parameter(torch.randn(num_heads,num_heads) * 0.01)
        self.gate = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        

    def forward(self, x):
        """
        x: [B, T, 2, H]  0: real, 1: imag
        """
        B, T, _, H = x.shape
        d = self.head_dim
        #if self.causal_mask is None or self.causal_mask.size(0) != T:
        mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).detach()  # [T, T]
        causal_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        

        # 1. take real part to build attentions，multi_head
        z = x[:, :, 0, :]  # [B, T, H]
        z = z.view(B, T, self.num_heads, d)  # [B, T, num_heads, d]

        # 2. compute the attention score
        zB = torch.einsum('bthd, hdd -> bthd', z, self.B)  # [B, T, num_heads, d]
        attn_scores = torch.einsum('bthd, bHhd -> bthH', zB, z)  # [B, T, num_heads, T]
        #attn_scores = torch.sigmoid(attn_scores)
        
        
        # 3. extract sub diag （for tree penalty）
        subdiag = torch.diagonal(attn_scores, offset=1, dim1=1, dim2=3)  # [B, num_heads, T-1] 
        #subdiag = -subdiag
        #assert not torch.any(subdiag < 0), f"subdiag has unsuitable values! "
        #assert not torch.any(subdiag > 1), f"subdiag has unsuitable values! "
        # 4. tree penaly construct using accum differ matrix (log sum exp)!
        accu = torch.cumsum(torch.exp(-subdiag), dim=-1)  # [B, num_heads, T-1]
        accu = torch.cat([torch.zeros(B, self.num_heads, 1, device=x.device), accu], dim=-1)  # [B, num_heads, T]
        #print(accu[0,0,:])
        C = accu.unsqueeze(-1) - accu.unsqueeze(-2)  # [B, num_heads, T, T]# [B, num_heads, T, T]
        C = C.masked_fill(causal_mask == 0, 0)
        assert not torch.any(C < 0), f"C has negative values! min: {C.min().item()}"
        C = torch.log(1+C+1e-6)
        assert not torch.any(C < 0), f"C has negative values after log! min: {C.min().item()}"
        # 5. apply penalty
        #attn_scores = torch.sigmoid(attn_scores)
        
        attn_scores = attn_scores.permute(0, 2, 1, 3)  # [B, num_heads, T, T]
        attn_scores = 1 + F.relu(attn_scores - self.p * C)

        # 6. cross head attention!
        sim_maxsubdiag = C[:,:,-1,0] # [B, num_heads]  sim maxium scores in subdiag selected!
        zBF = torch.einsum('bh, hh -> bh', sim_maxsubdiag, self.B_FUSE)  # [B, num_heads]
        cross_attn = torch.einsum('bh, bH -> bhH', zBF, sim_maxsubdiag)  # [B, num_heads, num_heads]
        cross_attn = 1+F.relu(cross_attn)
        cross_attn = cross_attn / (torch.sum(cross_attn, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, num_heads]

        # update attn_scores
        attn_scores = torch.einsum('bhtT, bhh -> bhtT', attn_scores, cross_attn)  # [B, num_heads, T, T]

        # causual mask + normalize
        attn_scores = attn_scores.masked_fill(causal_mask == 0, 0)
        attn_scores = attn_scores / (torch.sum(attn_scores, dim=-1, keepdim=True) + 1e-6)  # [B, num_heads, T, T]
       
        

        # average each head's score after per head attention
        attn_final = attn_scores.mean(dim=1)  # [B, T, T]
        #attn_final = F.silu(self.B_C(attn_scores).squeeze(-1))  # [B, T, T]

        # 7. update x
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

        # gate
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
        # ★ reverse assoc for penalty
        assoc = torch.sigmoid(-assoc)  # compress to  (0,1)，val reorder


        C = torch.zeros(B, T, T, device=h_real.device)
        for i in range(T):
            
            C[:,i,i] = 0
            max_val = torch.full((B,), -1e6, device=h_real.device)
            for j in range(i+1, T):
                max_val = torch.max(max_val, assoc[:, j-1])
                C[:, j, i] = max_val

        
        return C

    def forward(self, x):
        B, T,_ ,H = x.shape

        #B_pos = torch.matmul(self.B, self.B.T)

        h_real_input = x[:,:,0,:]
        h_imag_input = x[:,:,1,:]

        attn_scores = torch.matmul( torch.matmul(h_real_seq, self.B),  # [B, T, H]
                                    h_real_seq.transpose(1, 2)        # [B, H, T]
                                                                    )  # [B, T, T]
        attn_scores = torch.sigmoid(attn_scores)
        # ★ implement tree penalty
        C = self.tree_penalty(h_real_seq)  # [B, T, T] 

        attn_scores = attn_scores - self.z * C
        attn_scores = 1+F.relu(attn_scores)
        mask = torch.tril(torch.ones(T, T, device=h_real_seq.device), diagonal=0).detach()  # [T, T]
        mask = mask.unsqueeze(0)  # [1, T, T]
        attn_scores = attn_scores.masked_fill(mask == 0, 0)
        
        attn_weights = attn_scores / (attn_scores.sum(dim=1, keepdim=True) + 1e-6) # [B, T, T]

        # ★ apply attention ：attn_weights @ h_real_seq
        h_real_recur = torch.matmul(attn_weights, h_real_seq)  # [B, T, H]
        h_imag_recur = torch.matmul(attn_weights, h_imag_seq)  # [B, T, H]
        
        gate = torch.sigmoid(self.skip_gate).unsqueeze(0).unsqueeze(0)
        h_real_skip = h_real_input * gate + F.silu(h_real_recur) 
        h_imag_skip = h_imag_input * gate + F.silu(h_imag_recur)

        magnitude = torch.sqrt(h_real_skip**2 + h_imag_skip**2 + 1e-8)
        h_real_out = h_real_skip / magnitude
        h_imag_out = h_imag_skip / magnitude

        return torch.stack([h_real_out, h_imag_out],dim=2)
    
    


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

    def forward(self, x):
        """
        Args:
            h_real_seq, h_imag_seq: [B, T, H]
        Returns:
            h_real_out, h_imag_out: [B, T, H]
        """
        B, T,_, H = x.shape

        # Save input for skip connection
        h_real_input = x[:,:,0,:]
        h_imag_input = x[:,:,1,:]

        h_real_prev = torch.zeros(B, H, device=x.device)
        h_imag_prev = torch.zeros(B, H, device=x.device)
        #h_real_prev = h_real_input[:,-1,:]
        #h_imag_prev = h_imag_input[:,-1,:]
        
        outputs_real, outputs_imag = [], []

        for t in range(T):
            h_real_t = x[:, t, 0 , :]
            h_imag_t = x[:, t, 1 , :]

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

        return torch.stack([h_real_out, h_imag_out], dim=2)


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
        
        x = torch.stack([h_real,h_imag],dim=2)

        for layer in self.layers:
            x = layer(x)  #normed before layer output
        # Phase decoding: read out phase information
        phase = torch.atan2(x[:,-1,0,:], h_real[:,-1,1,:]) 
        #phase = self.masked_atan(h_imag[:,-1,:], h_real[:,-1,:])
        x_dec  =  torch.cat([torch.cos(phase),torch.sin(phase)],dim=-1) 
        #x = torch.stack([h_imag[:,-1,:],h_real[:,-1,:]],dim=-1)
        
        #x = torch.einsum('bhc, hc -> bh', x, self.c_weight) + self.c_bias
        #warp_unwarp_phase_last = unwarp_phase[:, -1, :]
        #return self.decoder(warp_unwarp_phase_last)
        return self.decoder(x_dec)

class CSP_BLOCK(nn.Module):
    def __init__(self, head_dim=16, n_head=4, num_layers=3, model_mode='atten_mhead'): 
        super().__init__()
        self.hidden_dim = head_dim*n_head

        if model_mode == 'atten':
            self.layers = nn.ModuleList([ComplexPRLayer_ATTENTION(self.hidden_dim) for _ in range(num_layers)])
            print("Atten mode received! I'm createing attention style hidden layer!")
        elif model_mode == 'atten_mhead':
            self.layers = nn.ModuleList([LinearTreeAttnLayer(n_head, self.hidden_dim, p=0.1) for _ in range(num_layers)])
            print("Linear Tree Atten mode received! I'm createing tree attention style hidden layer!")
        elif model_mode == "atten_rbf":
            self.layers = nn.ModuleList([LinearMhRBFKAttnLayer(n_head, self.hidden_dim, p=0.1) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([ComplexPRLayer(self.hidden_dim) for _ in range(num_layers)])
            print("No known mode explicted! Draw back to relax mode with shared weights and per domain activate!")

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class CSP_Seq2Seq(nn.Module):
    def __init__(self, vocab_size, head_dim=16, n_head=4,num_layers=3,embed_dim=32,sos_idx=-4,model_mode='atten_mhead',train_method='teacher_forcing',angle_step =0.0025):
        super().__init__()
        self.train_method = train_method
        self.vocab_size = vocab_size
        self.hidden_dim = head_dim*n_head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_proj = nn.Linear(embed_dim, self.hidden_dim)
        #self.phase_proj   = nn.Linear(2*self.hidden_dim,self.hidden_dim)
        self.blocks   = nn.ModuleList([CSP_BLOCK(head_dim,n_head,num_layers,model_mode)])
        #self.decision_kernel = nn.Parameter(torch.randn(embed_dim,embed_dim))
        self.output_proj  = lambda x: torch.matmul(x, self.embedding.weight.T)
        self.max_len = 30
        self.sos_idx=sos_idx
        self.decoder_proj = nn.Linear(2*self.hidden_dim,embed_dim)
        self.angle_step = angle_step
        
        self.alpha_proj = nn.Linear(self.hidden_dim,1)
        
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
        h = torch.tanh(self.encoder_proj(x))  #may be used in teacher forcing!
        h_real, h_imag = h, torch.zeros_like(h)
        h_real, h_imag = self.rotation(h_real,h_imag,angle_step=self.angle_step)
        x = torch.stack([h_real,h_imag],dim=2)
        

        for block in self.blocks:
            x = block(x)

        # extract phase for initial of decode
        '''
        phase = torch.atan2(x[:, :, 0,:], x[:, :, 1,:] + 1e-8)
        h_cos, h_sin = torch.cos(phase), torch.sin(phase)
        h_phasor = torch.cat([h_cos, h_sin], dim=-1)
        hidden_out = torch.tanh(self.phase_proj(h_phasor))  # [B, H] the last hidden state
        '''
        hidden_out = x[:,-1,:,:]
        if self.train_method == 'teacher_forcing':
            
            if target_ids is not None:
                T_out = target_ids.shape[1]
                out_embeds = self.embedding(target_ids)
                outputs = []
                current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)  # <SOS>
                emb = self.embedding(current_token)  # [B, E]
                out_embeds = torch.cat([emb.unsqueeze(1),out_embeds],dim=1)
                emb_hidden = torch.tanh(self.encoder_proj(out_embeds))
                emb_hr, emb_hi = self.rotation(emb_hidden, torch.zeros_like(emb_hidden),self.angle_step,None)
                emb_hidden = torch.stack([emb_hr, emb_hi],dim=-2)
                for t in range(T_out):
                    alpha      = torch.sigmoid(self.alpha_proj(hidden_out[:,0,:]))
                    hidden_out = torch.stack([alpha,alpha],dim=1)*hidden_out + F.silu(emb_hidden[:,t,:,:])  # 
                    phase = torch.atan2(hidden_out[:, 0,:], hidden_out[:, 1,:] + 1e-8)
                    h_cos, h_sin = torch.cos(phase), torch.sin(phase)
                    h_phasor = torch.cat([h_cos, h_sin], dim=-1)
                    #hidden_out = torch.tanh(self.phase_proj(h_phasor))
                    emb_out = torch.tanh(self.decoder_proj(h_phasor))
                    #torch.matmul(self.embedding.weight, self.decision_kernel
                    logits = self.output_proj(emb_out)
                    outputs.append(logits)
                    emb = out_embeds[:, t, :]  # [B, E]
                    magnitude = torch.sqrt(hidden_out[:,0,:]**2 + hidden_out[:,1,:]**2 + 1e-8)
                    hidden_out = hidden_out/magnitude.unsqueeze(1)
                return torch.stack(outputs, dim=1)

            else:
                outputs = []
                current_token = torch.full((B,), self.sos_idx, device=input_ids.device, dtype=torch.long)
                pre_angle = -self.angle_step
                for t in range(40):
                    emb            = self.embedding(current_token)
                    emb_hidden     = torch.tanh(self.encoder_proj(emb))
                    emb_hr, emb_hi = self.rotation(emb_hidden, torch.zeros_like(emb_hidden),self.angle_step,pre_angle=pre_angle)
                    emb_hidden = torch.stack([emb_hr, emb_hi],dim=-2)
                    alpha      = torch.sigmoid(self.alpha_proj(hidden_out[:,0,:]))
                    hidden_out = torch.stack([alpha,alpha],dim=1)*hidden_out + F.silu(emb_hidden)
                    phase = torch.atan2(hidden_out[:, 0,:], hidden_out[:, 1,:] + 1e-8)
                    h_cos, h_sin = torch.cos(phase), torch.sin(phase)
                    h_phasor = torch.cat([h_cos, h_sin], dim=-1)
                    #hidden_out = torch.tanh(self.phase_proj(h_phasor))
                    emb_out = torch.tanh(self.decoder_proj(h_phasor))
                    logits = self.output_proj(emb_out)
                    next_token = torch.argmax(logits, dim=-1)
                    outputs.append(next_token)
                    current_token = next_token
                    pre_angle += self.angle_step
                    magnitude = torch.sqrt(hidden_out[:,0,:]**2 + hidden_out[:,1,:]**2 + 1e-8)
                    hidden_out = hidden_out/magnitude.unsqueeze(1)
                    
                return torch.stack(outputs, dim=1)
            
        else:     # using transformer style recurrent reasoning! train and test aligned!
         
            emb_out_norm   = F.normalize(hidden_out, p=2, dim=-1)  # [B, T, embed_dim]
            embedding_norm = F.normalize(self.embedding.weight, p=2, dim=-1)  # [vocab_size, embed_dim]

            # cosine logits（range [-1, 1]）
            logits = torch.matmul(emb_out_norm, embedding_norm.T)  # [B, T, vocab_size]
            if target_ids is not None:
                # training step：return logits directly
                return logits
            else:
                # reasoning： elect the maximum element index
                outputs = torch.argmax(logits, dim=-1)  # [B, T]
                return outputs

            
        
    
    

    
    
