import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== 1. causal mask ====================
def create_causal_mask(seq_len, device):
    """creating causal mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()  #


# ==================== 2. pos encoding（learnable） ====================
class LearnablePositionalEncoding(nn.Module):
    """learnable pos encoding（much better than consine waves）"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    
    def forward(self, x):
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


# ==================== 3. Decoder Layer（self regression） ====================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # self_attn（with causal mask）
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ==================== 4. Pure Decoder Transformer（Naming in sholarship：ARFormer） ====================
class ARFormer(nn.Module):
    """
    AutoRegressive Former (ARFormer)
    Decoder architecutre, seq generate using self regression
    
    normalize name：
    - AR: AutoRegressive (self regression)
    - Former: Short for Transformer
    - Names in the Literature: AR-Transformer, GPT-like, CausalFormer
    """
    def __init__(
        self,
        vocab_size,
        d_model=64,          # dimension of embedding
        num_heads=4,         # number of attention head
        num_layers=2,        # number of layers
        d_ff=128,            # FFN dim
        max_len=256,         # maximum sequence
        dropout=0.1,
        output_dim=10        # output dim（0-9）
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. pos encoding（learnable）
        self.pos_encoding = LearnablePositionalEncoding(max_len, d_model)
        
        # 3. Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Layer Norm (Pre-norm style,stable at training!)
        self.norm = nn.LayerNorm(d_model)
        
        # 5. output head（project to vocab）
        self.lm_head = nn.Linear(d_model, vocab_size)  # generate main sequence
        
        # 6. classifiy head（to predict the final ansers）
        self.classifier = nn.Linear(d_model, output_dim)
        
        # weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization,stable at training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_logits=True):
        """
        Args:
            x: [B, T] input sequence
            return_logits:  return logits if True, classify result if False
        
        Returns:
            logits: [B, T, vocab_size] if return_logits=True
            class_logits: [B, output_dim] if return_logits=False
        """
        B, T = x.shape
        
        # 1. Embedding
        x = self.token_embedding(x)  # [B, T, D]
        x = self.pos_encoding(x)
        
        # 2. causal mask（only useful at traning）
        mask = create_causal_mask(T, x.device)  # [T, T]
        
        # 3. pass each layer
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # 4. Layer Norm
        x = self.norm(x)
        
        if return_logits:
            # traning：predict the next token
            logits = self.lm_head(x)  # [B, T, vocab_size]
            return logits
        else:
            # reasoning：predict using the last time hidden 
            last_token = x[:, -1, :]  # [B, D]
            class_logits = self.classifier(last_token)  # [B, output_dim]
            return class_logits
    
    def generate(self, input_ids, max_new_tokens=50):
        """
        self regression（reasoning）
        
        Args:
            input_ids: [B, T] initialized sequence（如 "<SOS>"）
            max_new_tokens: maximum tokens to be generated
        
        Returns:
            generated: [B, T+max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # get logits of current sequence
                logits = self.forward(input_ids, return_logits=True)  # [B, T, vocab_size]
                # the the last time step
                next_token_logits = logits[:, -1, :]  # [B, vocab_size]
                #  greddy decode （also sampling）
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
                # concat the sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # stop if eos
                if (next_token == self.eos_token_id).all():
                    break
        return input_ids


# ==================== 5. 轻量级版本（参数量~119K） ====================
class MiniARFormer(ARFormer):
    """
    Mini AutoRegressive Former
    
    """
    def __init__(self, vocab_size, output_dim=10):
        super().__init__(
            vocab_size=vocab_size,
            d_model=64,          # 
            num_heads=4,         # 
            num_layers=2,        # 
            d_ff=128,            # 
            max_len=256,
            dropout=0.1,
            output_dim=output_dim
        )
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ MiniARFormer numel: {total_params:,}")


# ==================== 6. standard version（for scaling comparision） ====================
class ARFormerBase(ARFormer):
    """basic version, abut 500K param volumn"""
    def __init__(self, vocab_size, output_dim=10):
        super().__init__(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=8,
            num_layers=3,
            d_ff=256,
            max_len=256,
            dropout=0.1,
            output_dim=output_dim
        )



        
        
        
        
        
        
        

        
        
        
# ==================== 7. usages ====================
if __name__ == "__main__":
    vocab_size = 50  # containing digits 0-9, +, -, *, /, (, ), mod, =, repeat, <SOS>, <EOS>, <PAD>
    model = MiniARFormer(vocab_size=vocab_size, output_dim=10)
    
    # input
    batch_size, seq_len = 4, 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # train mode：peredict the next token
    logits = model(x, return_logits=True)
    print(f"shape at traning: {logits.shape}")  # [4, 32, 50]
    
    # reasoning：calssifies
    class_logits = model(x, return_logits=False)
    print(f"shape at reasoning: {class_logits.shape}")  # [4, 10]
