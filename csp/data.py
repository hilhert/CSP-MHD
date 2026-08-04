import torch
from torch.utils.data import DataLoader, TensorDataset



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

def create_dataloaders(X, y, batch_size=64, train_ratio=0.8):
    num_samples = X.shape[0]
    num_train = int(num_samples * train_ratio)
    indices = torch.randperm(num_samples)
    train_idx, test_idx = indices[:num_train], indices[num_train:]
    
    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[test_idx], y[test_idx]), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader