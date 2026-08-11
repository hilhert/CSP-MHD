import torch
import random
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
from abc import ABC, abstractmethod

class ArithmeticGenerator(ABC):
    """算术表达式生成器基类，支持不同运算类型"""
    
    @abstractmethod
    def generate(self, max_terms=3, max_digits=2, min_val=0, max_val=100):
        pass

def generate_parity_data(num_samples=5000, seq_len=16):
    X = torch.randint(0, 2, (num_samples, seq_len)).float().unsqueeze(-1)
    y = (X.squeeze(-1).sum(dim=1) % 2).long()
    return X, y

def generate_mod3_data(num_samples=5000, seq_len=16):
    X = torch.randint(0, 2, (num_samples, seq_len)).float().unsqueeze(-1)
    ones_count = X.squeeze(-1).sum(dim=1)
    y = (ones_count % 3 != 0).long()  # 能被3整除 -> 0，否则 -> 1
    return X, y
def generate_parenthesis_data(num_samples=5000, seq_len=16):
    """
    generate parthesis data

    input: [num_samples, seq_len, 1]
        - 0 represents '('
        - 1 represents ')'
    label: [num_samples]
        - 0: unmathced
        - 1: matched
    """
    X = []
    y = []

    for _ in range(num_samples):
        # generate binary sequence with seq_len 
        seq = torch.randint(0, 2, (seq_len,)).tolist()

        # 检查括号是否匹配
        balance = 0
        is_valid = True
        for val in seq:
            if val == 0:  # '('
                balance += 1
            else:         # ')'
                balance -= 1
                if balance < 0:  #  ')' is more than '('
                    is_valid = False
                    break

        #  balance must be 0 at final
        if is_valid and balance == 0:
            y.append(1)
        else:
            y.append(0)

        X.append(seq)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.long)

    return X, y





class ModNArithmeticGenerator(ArithmeticGenerator):
    """
    模 N 算术表达式生成器
    
    生成形如 "3 + 5 = 8 mod 10" 或 "7 - 2 = 5 mod 3" 的表达式
    用于 seq2seq 训练，帮助模型学习模运算规则
    """
    
    def __init__(self, n=10):
        """
        Args:
            n: 模数，默认为 10
        """
        self.n = n
        self._operators = ['+', '-']
    
    def generate(self, max_terms=3, max_digits=2, min_val=0, max_val=100):
        """
        生成一个模 N 算术表达式
        
        Returns:
            (expr, result): 表达式字符串和结果字符串
        """
        num_terms = random.randint(2, max_terms)
        tokens = []
        total = 0
        
        for i in range(num_terms):
            num = random.randint(0, self.n - 1)
            if i == 0:
                total = num
            else:
                op = random.choice(self._operators)
                if op == '+':
                    total += num
                else:
                    total -= num
                tokens.append(op)
            tokens.append(str(num))
        
        # 应用模运算
        result = total % self.n
        expr = ' '.join(tokens)
        return expr + ' = ', str(result)
    
    def build_vocab(self):
        """构建模 N 运算的词表"""
        digits = [str(i) for i in range(self.n)]
        return digits + ['+', '-', '=', ' ', '<SOS>', '<EOS>', '<PAD>']
    
    def __call__(self, *args, **kwargs):
        """使实例可调用，兼容 Dataset 接口"""
        return self.generate(*args, **kwargs)



def create_dataloaders(X, y, batch_size=64, train_ratio=0.8):
    num_samples = X.shape[0]
    num_train = int(num_samples * train_ratio)
    indices = torch.randperm(num_samples)
    train_idx, test_idx = indices[:num_train], indices[num_train:]
    
    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[test_idx], y[test_idx]), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



class SymbolicArithmeticDataset(Dataset):
    def __init__(self, num_samples, max_terms=3, max_digits=2, min_val=0, max_val=100,
                 generate_expression_func=None, vocab=None,mode='copy'):
        """
        Args:
            generate_expression_func: 生成器实例或函数，如 ModNArithmeticGenerator(n=7)
            vocab: 预构建的词表，如果为 None 则从生成器推导
        """
        self.mode = mode
        self._generate_expression = generate_expression_func
        
        # 生成样本
        self.samples = self._generate(num_samples, max_terms, max_digits, min_val, max_val)
        
        # 词表构建：优先使用传入的 vocab，否则从生成器推导
        if vocab is not None:
            self.vocab = vocab
        elif hasattr(generate_expression_func, 'build_vocab'):
            # 如果生成器类提供了 build_vocab 方法（如 ModNArithmeticGenerator）
            self.vocab = generate_expression_func.build_vocab()
        else:
            self.vocab = self._build_vocab_from_samples()
        
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.max_len = max(len(s['input']) for s in self.samples) + 10

    def _build_vocab_from_samples(self):
        vocab_set = set()
        for s in self.samples:
            vocab_set.update(s['input'])
            vocab_set.update(s['output'])
        vocab_set.update(['<SOS>', '<EOS>', '<PAD>'])
        # 排序保持稳定，数字优先，特殊 token 放最后
        digits = [c for c in vocab_set if c.isdigit()]
        others = [c for c in vocab_set if not c.isdigit() and c not in ['<SOS>', '<EOS>', '<PAD>']]
        specials = ['<SOS>', '<EOS>', '<PAD>']
        return sorted(digits) + sorted(others) + specials

    def _build_vocab(self):
        # 保留旧版兼容性
        return self._build_vocab_from_samples()

    def _generate(self, num_samples, max_terms, max_digits, min_val, max_val):
        samples = []
        for _ in range(num_samples):
            expr, result = self._generate_expression(max_terms, max_digits, min_val, max_val)
            samples.append({'input': expr , 'output': result})
            
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        #print(f"expr: '{sample['input']}', result: '{sample['output']}'")  # ★ 加这行
        input_tokens = [self.char2idx[c] for c in sample['input']]
        output_tokens = [self.char2idx[c] for c in sample['output']]

        input_ids =  [self.char2idx['<SOS>']] + input_tokens 
        if self.mode == 'copy':
            # 复制模式：输出 = 输入内容 + <EOS>
            output_ids = input_tokens + [self.char2idx['<EOS>']]
        elif self.mode == 'complete':
            # 复制+补全模式：输出 = 输入内容 + 结果 + <EOS>
            output_ids = input_tokens + output_tokens + [self.char2idx['<EOS>']]
        else:  # compute
            output_ids = output_tokens + [self.char2idx['<EOS>']]
        
        # 有效长度（不含 padding）
        in_len = len(input_ids)
        out_len = len(output_ids)
        
        input_ids = input_ids + [self.char2idx['<PAD>']] * (self.max_len - in_len)
        output_ids = output_ids + [self.char2idx['<PAD>']] * (self.max_len - out_len)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'in_len':  in_len,
            'out_len': out_len
        }
    
    def __len__(self):
        return len(self.samples)