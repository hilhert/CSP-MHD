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
    
def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
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
    checkpoint = torch.load(filepath, map_location=device)
    
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

def train_model_seq(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        output_ids = batch['output_ids'].to(device)
        out_len = batch['out_len'].to(device)
        
        # 直接传 output_ids，forward 内部会处理
        logits = model(input_ids, output_ids)  # [B, T, vocab_size]
        targets = output_ids  # [B, T]
        
        mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < out_len.unsqueeze(1)
        
        loss = criterion(logits.permute(0, 2, 1), targets)
        loss = (loss * mask.float()).sum() / mask.sum()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_seq(model, dataloader, device, pad_idx, eos_idx, debug=True):
    model.eval()
    correct = 0
    total = 0
    idx2char = dataloader.dataset.idx2char
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            out_len = batch['out_len']
            
            pred_ids = model(input_ids, target_ids=None)  # [B, T]
            
            for i in range(len(pred_ids)):
                # 直接用 out_len 截取，不需要再跳过 <SOS>
                pred_tokens = pred_ids[i, :out_len[i]].tolist()
                target_tokens = output_ids[i, :out_len[i]].tolist()
                
                pred_tokens = [x for x in pred_tokens if x not in [pad_idx, eos_idx]]
                target_tokens = [x for x in target_tokens if x not in [pad_idx, eos_idx]]
                
                pred_str = ''.join(idx2char[x] for x in pred_tokens)
                target_str = ''.join(idx2char[x] for x in target_tokens)
                
                if debug and batch_idx == 0 and i < 5:
                    input_tokens = batch['input_ids'][i].tolist()
                    input_tokens = [x for x in input_tokens if x not in [pad_idx, eos_idx]]
                    input_str = ''.join(idx2char[x] for x in input_tokens)
                    print(f"[Debug] Input: {input_str}")
                    print(f"[Debug] Pred : {pred_str}")
                    print(f"[Debug] Target: {target_str}")
                    print(f"pred_ids[0]: {pred_ids[0].tolist()}")
                    print("-" * 40)
                
                if pred_str == target_str:
                    correct += 1
                total += 1
    
    return correct / total


def train_epoch_seq(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        output_ids = batch['output_ids'].to(device)
        out_len = batch['out_len']
        
        # 直接传 output_ids，forward 内部会处理
        logits = model(input_ids, output_ids)  # [B, T, vocab_size]
        targets = output_ids  # [B, T]
        
       # 基础掩码：有效位置为 True
        mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < out_len.unsqueeze(1)

        # 构造权重矩阵：有效位置权重为 1，最后一个有效 token 权重为 out_len - 1
        weight = mask.float()
        '''
        for i in range(targets.size(0)):
            weight[i, out_len[i] - 1] = out_len[i] - 1  # 只对每个样本的最后一个有效 token 设置权重
        '''
        loss = criterion(logits.permute(0, 2, 1), targets)
        loss = (loss * weight).sum() / weight.sum()
        
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)