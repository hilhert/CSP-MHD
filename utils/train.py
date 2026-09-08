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
    # save .pt checkpoint
    torch.save(checkpoint, filepath)
    
    # save safetensors
    base_path, _ = os.path.splitext(filepath)
    safetensors_path = base_path + ".safetensors"
    
    save_model(model, safetensors_path)
    
def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    resotre traning from .pt or .safetensors
    
    params:
        filepath: .pt location of checkpoint (i.e "checkpoints/best_model.pt")
        model: the initialized PyTorch model
        optimizer: the initialized PyTorch optimizer (optional)
        
    return:
        model, optimizer, start_epoch, f1
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"could not find model file: {filepath}")
        
    print(f"[Checkpoint] restord checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    # 1. restore model weight
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. restore optimizer state (if saved)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1  # continue from next epoch
    loss = checkpoint.get('loss', 0.0)
    f1 = checkpoint.get('f1', 0.0)
    
    print(f"[Checkpoint] restored at  Epoch {start_epoch} | 上次 Loss: {loss:.4f} | F1: {f1:.4f}")
    return model, optimizer, start_epoch, f1


def load_model(filepath, model, device='cpu'):
    """
    [reason/evaluate] load model weight
    load .safetensors firstly，read .pt weight if safetensor not exists
    
    param:
        filepath: file path of directory path (i.e "best_model.safetensors" 或 "best_model.pt")
        model:  PyTorch model which have been initialized
        device: object device ('cpu', 'cuda' ...)
        
    return:
        model ( model.eval() state)
    """
    base_path, _ = os.path.splitext(filepath)
    safetensors_path = base_path + ".safetensors"
    pt_path = base_path + ".pt"
    
    model = model.to(device)
    
    # 优先寻找 .safetensors 加载
    if os.path.exists(safetensors_path):
        print(f"[Model] using Safetensors format to load weights: {safetensors_path}")
        st_load_model(model, safetensors_path)
    # 次选加载 .pt 中的权重
    elif os.path.exists(pt_path):
        print(f"[Model] could not find Safetensors，draw back to  PyTorch .pt format: {pt_path}")
        checkpoint = torch.load(pt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"could not find checkpoint of safetensors (.safetensors 或 .pt): {base_path}")
        
    model.eval()  # switch to eval
    print(f"[Model] model weight loaded successfully，ready to eval！")
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

def train_model_seq(model, dataloader, optimizer, criterion, device,focal=False):
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
        
        
        
        
        if focal:
            '''
            max_len = loss.size(1)
            
            t = torch.arange(max_len, device=loss.device).float().unsqueeze(0)  # [1, T]
            # using sigmoid to project val range to [0.5,1.0]
            weights = torch.sigmoid((t - max_len/2) / (max_len/6))  # [1, T]
            
            weights = weights.repeat(loss.size(0), 1)  # [B, T]
            
            weights = weights * mask.float()
            
            loss = (loss * weights).sum()/ (weights.sum() + 1e-6)
            '''
            head_weight = 3
            #tail_weight = 2.0
            seq_weight = 1

            # head（the first token）is the answer
            head_loss = criterion(logits[:, 0, :], targets[:, 0])
            seq_loss = criterion(logits[:, 1:, :].permute(0,2,1), targets[:, 1:])

            loss = (seq_weight * seq_loss + head_weight * head_loss)/(head_weight+seq_weight) 
           

        else:    
            loss = criterion(logits.permute(0, 2, 1), targets)
        
        loss = (loss * mask.float()).sum() / mask.sum()    
        total_norm = 0.0
        for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader), total_norm


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

'''
def train_epoch_seq(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        output_ids = batch['output_ids'].to(device)
        out_len = batch['out_len']
        
        # pass output_ids，forward function handle it
        logits = model(input_ids, output_ids)  # [B, T, vocab_size]
        targets = output_ids  # [B, T]
        
       # basic mask： True at valid position
        mask = torch.arange(targets.size(1), device=device).unsqueeze(0) < out_len.unsqueeze(1)

        # construct weight matrix：1 at valid pos，the last token weight should be out_len - 1
        weight = mask.float()
        
        for i in range(targets.size(0)):
            weight[i, out_len[i] - 1] = out_len[i] - 1  # 
        
        loss = criterion(logits.permute(0, 2, 1), targets)
        loss = (loss * weight).sum() / weight.sum()
        
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
'''