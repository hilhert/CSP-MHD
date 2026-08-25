import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== 1. 因果注意力掩码 ====================
def create_causal_mask(seq_len, device):
    """生成上三角掩码，防止看到未来信息"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()  # True表示被mask掉的位置


# ==================== 2. 位置编码（可学习） ====================
class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码（比正弦波更灵活）"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    
    def forward(self, x):
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


# ==================== 3. 纯Decoder层（自回归） ====================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 自注意力（带因果掩码）
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


# ==================== 4. 纯Decoder Transformer（学术命名：ARFormer） ====================
class ARFormer(nn.Module):
    """
    AutoRegressive Former (ARFormer)
    纯Decoder架构，用于自回归序列生成
    
    命名规范：
    - AR: AutoRegressive (自回归)
    - Former: Transformer的缩写
    - 学术论文中常用: AR-Transformer, GPT-like, CausalFormer
    """
    def __init__(
        self,
        vocab_size,
        d_model=64,          # 嵌入维度
        num_heads=4,         # 注意力头数
        num_layers=2,        # 层数
        d_ff=128,            # FFN中间维度
        max_len=256,         # 最大序列长度
        dropout=0.1,
        output_dim=10        # 输出维度（数字0-9）
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码（可学习）
        self.pos_encoding = LearnablePositionalEncoding(max_len, d_model)
        
        # 3. Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Layer Norm (Pre-norm风格，训练更稳定)
        self.norm = nn.LayerNorm(d_model)
        
        # 5. 输出头（映射到词表）
        self.lm_head = nn.Linear(d_model, vocab_size)  # 用于生成序列
        
        # 6. 分类头（用于最终答案预测）
        self.classifier = nn.Linear(d_model, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化，让训练更稳定"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_logits=True):
        """
        Args:
            x: [B, T] 输入序列
            return_logits: True返回词表logits，False返回分类结果
        
        Returns:
            logits: [B, T, vocab_size] 如果return_logits=True
            class_logits: [B, output_dim] 如果return_logits=False
        """
        B, T = x.shape
        
        # 1. Embedding
        x = self.token_embedding(x)  # [B, T, D]
        x = self.pos_encoding(x)
        
        # 2. 因果掩码（只对训练时用）
        mask = create_causal_mask(T, x.device)  # [T, T]
        
        # 3. 经过Decoder层
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # 4. Layer Norm
        x = self.norm(x)
        
        if return_logits:
            # 用于训练：预测下一个token
            logits = self.lm_head(x)  # [B, T, vocab_size]
            return logits
        else:
            # 用于推理：取最后一个时间步分类
            last_token = x[:, -1, :]  # [B, D]
            class_logits = self.classifier(last_token)  # [B, output_dim]
            return class_logits
    
    def generate(self, input_ids, max_new_tokens=50):
        """
        自回归生成（推理时用）
        
        Args:
            input_ids: [B, T] 初始序列（如 "<SOS>"）
            max_new_tokens: 最多生成多少个新token
        
        Returns:
            generated: [B, T+max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 获取当前序列的logits
                logits = self.forward(input_ids, return_logits=True)  # [B, T, vocab_size]
                # 取最后一个时间步
                next_token_logits = logits[:, -1, :]  # [B, vocab_size]
                # 贪心解码（也可以用采样）
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
                # 拼接到序列
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 如果生成了EOS，提前停止
                if (next_token == self.eos_token_id).all():
                    break
        return input_ids


# ==================== 5. 轻量级版本（参数量~119K） ====================
class MiniARFormer(ARFormer):
    """
    Mini AutoRegressive Former
    专门优化到119K参数，用于公平对比
    """
    def __init__(self, vocab_size, output_dim=10):
        super().__init__(
            vocab_size=vocab_size,
            d_model=64,          # 小维度
            num_heads=4,         # 4个头
            num_layers=2,        # 只有2层（depth小一点）
            d_ff=128,            # FFN维度
            max_len=256,
            dropout=0.1,
            output_dim=output_dim
        )
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ MiniARFormer 参数量: {total_params:,}")


# ==================== 6. 标准版本（用于对比scaling效果） ====================
class ARFormerBase(ARFormer):
    """基础版，参数量约500K"""
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



        
        
        
        
        
        
        

        
        
        
# ==================== 7. 使用示例 ====================
if __name__ == "__main__":
    vocab_size = 50  # 包括数字0-9, +, -, *, /, (, ), mod, =, repeat, <SOS>, <EOS>, <PAD>
    model = MiniARFormer(vocab_size=vocab_size, output_dim=10)
    
    # 模拟输入
    batch_size, seq_len = 4, 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 训练模式：预测下一个token
    logits = model(x, return_logits=True)
    print(f"训练输出形状: {logits.shape}")  # [4, 32, 50]
    
    # 推理模式：分类
    class_logits = model(x, return_logits=False)
    print(f"推理输出形状: {class_logits.shape}")  # [4, 10]
