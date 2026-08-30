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





class ArithmeticGenerator(ABC):
    @abstractmethod
    def generate(self, max_terms=3, max_digits=2, min_val=0, max_val=100):
        pass


class ModNArithmeticGenerator(ArithmeticGenerator):
    """
    模 N 算术表达式生成器，支持括号嵌套，mod n 放在等号左侧
    """
    
    def __init__(self, n=None, simple=True, max_depth=2):
        self.fixed_n = n
        self.simple = simple
        self.max_depth = max_depth
        self._operators = ['+', '-']
        self._mod_range = (2, 10)
    
    def _get_mod_value(self):
        if self.fixed_n is not None:
            return self.fixed_n
        return random.randint(self._mod_range[0], self._mod_range[1])
    

    
    def _generate_expression(self, depth=0, n=10, wrap_outer=False):
        """递归生成表达式，支持括号嵌套"""
        if depth < self.max_depth and random.random() < 0.3:
            left = self._generate_expression(depth + 1, n)
            op = random.choice(self._operators)
            right = self._generate_expression(depth + 1, n)
            expr = f"({left} {op} {right})"
        else:
            num = random.randint(1-n, n - 1)
            return str(num)
            
        
        # 如果外层需要括号且当前表达式不是单独的叶子节点，可以加括号
        if wrap_outer and random.random() < 0.4 and depth <=1:
            return f"({expr})"
        return expr
    
    def _generate_sequence(self, max_terms, depth=0, n=10):
        """生成包含多个 term 的序列"""
        if max_terms == 0:
            return self._generate_expression(depth, n)
        if max_terms == 1:
            return self._generate_expression(depth, n)
        
        if random.random() < 0.2:
            num_terms = random.randint(2, max_terms)
            parts = []
            for i in range(num_terms):
                parts.append(self._generate_expression(depth + 1, n))
                if i < num_terms - 1:
                    parts.append(random.choice(self._operators))
            expr = ' '.join(parts)
            return f"({expr})"
        else:
            num_terms = random.randint(2, max_terms)
            parts = []
            for i in range(num_terms):
                parts.append(self._generate_expression(depth, n))
                if i < num_terms - 1:
                    parts.append(random.choice(self._operators))
            return ' '.join(parts)
    
    def generate(self, max_terms=3, max_digits=2, min_val=0, max_val=100):
        """生成模 N 算术表达式"""
        n = self._get_mod_value()
        if self.simple:
            return self._generate_simple(max_terms,n)
        else:
            # 生成表达式
            
            expr = self._generate_sequence(random.randint(0,max_terms), 0, n)

            # ★ 随机决定是否给最外层加括号（可选）
            if random.random() < 0.3:
                expr = f"({expr})"

            # 安全计算表达式值
            try:
                allowed_chars = set('0123456789+-*/() ')
                if not all(c in allowed_chars for c in expr):
                    raise ValueError("Expression contains invalid characters")
                total = int(eval(expr))
            except Exception:
                return self._generate_simple(max_terms, n)

            result = total % n
            return f"{expr} mod {n} = ", str(result), n
    
    def _generate_simple(self, max_terms, n):
        """简单模式（无括号）的回退方案"""
        num_terms = random.randint(2, max_terms)
        tokens = []
        total = 0
        for i in range(num_terms):
            num = random.randint(0, n - 1)
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
        result = total % n
        expr = ' '.join(tokens)
        if random.random() < 0.3:
            expr = f"({expr})"
        return f"{expr} mod {n} = ", str(result), n
    
    def build_vocab(self):
        """★ 固定词表：0-9 全部包含，不依赖模数"""
        digits = [str(i) for i in range(10)]
        return digits + ['+', '-', '=', ' ', '(', ')', '<SOS>', '<EOS>', '<PAD>', 'mod',' repeat ']
    
    def __call__(self, *args, **kwargs):
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

    def update(self,num_samples, max_terms, max_digits, min_val, max_val,generate_experssion_func=None ):
        
        if not generate_experssion_func:
            
            pass
        
        else:
            
            self._generate_expression = generate_expression_func
        
        self.samples = self._generate(num_samples, max_terms, max_digits, min_val, max_val)
            
        
    
    
    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            # 从最长匹配开始尝试
            for j in range(len(text), i, -1):
                if text[i:j] in self.char2idx:
                    tokens.append(text[i:j])
                    i = j
                    break
            # 如果没有匹配到，就跳过当前字符（理论上不会发生）
            else:
                i += 1
        return tokens
    
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
            expr, result,_ = self._generate_expression(max_terms, max_digits, min_val, max_val)
            samples.append({'input': expr , 'output': result})
            
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_tokens = [self.char2idx[c] for c in self.tokenize(sample['input'])]
        output_tokens = [self.char2idx[c] for c in self.tokenize(sample['output'])]
        #print("input: "+sample['input'])
        #print("output: "+sample['output'])
        input_ids = [self.char2idx['<SOS>']] + input_tokens

        if self.mode == 'copy':
            # 复制模式：输入内容 + <EOS>
            output_ids =  input_tokens + [self.char2idx['<EOS>']]
        elif self.mode == 'complete':
            # 复制+补全模式：输入内容 + 结果 + <EOS>
            output_ids = output_tokens + [self.char2idx[' repeat ']] + input_tokens + output_tokens + [self.char2idx['<EOS>']] 
        else:  # compute
            # 计算模式：结果 + <EOS>
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