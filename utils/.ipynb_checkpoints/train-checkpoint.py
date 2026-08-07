import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from safetensors.torch import save_model, save_file
import os

def save_checkpoint(model, optimizer, epoch, loss, f1, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'f1': f1,
    }
    # 保存 .pt 训练全量状态
    torch.save(checkpoint, filepath)
    
    # 剥离可能存在的后缀，确保导出 pure_name.safetensors
    base_path, _ = os.path.splitext(filepath)
    safetensors_path = base_path + ".safetensors"
    
    save_model(model, safetensors_path)
    
def load_checkpoint(filepath, model, optimizer=None):
    """
    【恢复训练专用】从 .pt 检查点加载模型与优化器状态。
    
    参数:
        filepath: .pt 检查点文件路径 (例如 "checkpoints/best_model.pt")
        model: 已初始化的 PyTorch 模型对象
        optimizer: 已初始化的 PyTorch 优化器对象 (可选)
        
    返回:
        model, optimizer, start_epoch, f1
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到检查点文件: {filepath}")
        
    print(f"[Checkpoint] 正在恢复训练状态: {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # 1. 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. 恢复优化器状态 (如果传入了 optimizer)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1  # 下轮继续
    loss = checkpoint.get('loss', 0.0)
    f1 = checkpoint.get('f1', 0.0)
    
    print(f"[Checkpoint] 成功恢复至 Epoch {start_epoch} | 上次 Loss: {loss:.4f} | F1: {f1:.4f}")
    return model, optimizer, start_epoch, f1


def load_model(filepath, model, device='cpu'):
    """
    【推理/评估专用】加载模型权重。
    优先加载 .safetensors，如果不存在则自动回退读取 .pt 权重。
    
    参数:
        filepath: 模型路径或基础路径 (例如 "best_model.safetensors" 或 "best_model.pt")
        model: 已初始化的 PyTorch 模型对象
        device: 目标设备 ('cpu', 'cuda' 等)
        
    返回:
        model (处于 model.eval() 状态)
    """
    base_path, _ = os.path.splitext(filepath)
    safetensors_path = base_path + ".safetensors"
    pt_path = base_path + ".pt"
    
    model = model.to(device)
    
    # 优先寻找 .safetensors 加载
    if os.path.exists(safetensors_path):
        print(f"[Model] 正在使用 Safetensors 加载模型权重: {safetensors_path}")
        st_load_model(model, safetensors_path)
    # 次选加载 .pt 中的权重
    elif os.path.exists(pt_path):
        print(f"[Model] 未找到 Safetensors，回退使用 PyTorch .pt 加载: {pt_path}")
        checkpoint = torch.load(pt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"未找到对应的权重文件 (.safetensors 或 .pt): {base_path}")
        
    model.eval()  # 自动切换为评估模式
    print(f"[Model] 模型权重加载完成，已就绪 (eval 模式)！")
    return model    


def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            _, pred_class = torch.max(pred, 1)
            correct += (pred_class == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


def evaluate_model_f1(model, test_loader, device='cpu'):
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            _, pred_class = torch.max(pred, 1)
            all_preds.extend(pred_class.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    return f1_score(all_targets, all_preds, average='binary')


def train_model(model, train_loader, test_loader, epochs=300,weight_decay=1e-8,lr=0.001, device='cpu', logger=None, loss_fn=None, cp_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*1.5, eta_min=1e-5)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    train_losses, f1s,test_accs, gradient_norms = [],[],[], []
    best_f1 = 0.0
    log = logger.info if logger else print
    
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        gradient_norms.append(total_norm)
        
        if epoch % 10 == 0:
            test_acc = evaluate_model(model, test_loader, device)
            f1       = evaluate_model_f1(model, test_loader, device)
            test_accs.append(test_acc)
            f1s.append(f1)
            #scheduler.step(test_acc)
            if  f1 > best_f1:
                save_checkpoint(model, optimizer, epoch, loss, f1, cp_path)
                best_f1 = f1
            log(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {test_acc:.4f} | F1: {f1:.4f} | GradNorm: {total_norm:.4f} | Best_F1: {best_f1:.4f}")
    
    return train_losses, f1s,test_accs, gradient_norms