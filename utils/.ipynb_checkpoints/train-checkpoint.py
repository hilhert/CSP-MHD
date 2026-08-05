import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

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


def train_model(model, train_loader, test_loader, epochs=300, lr=0.0003, device='cpu', logger=None, loss_fn=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    train_losses, test_accs, gradient_norms = [], [], []
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
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        gradient_norms.append(total_norm)
        
        if epoch % 10 == 0:
            test_acc = evaluate_model(model, test_loader, device)
            f1       = evaluate_model_f1(model, test_loader, device)
            test_accs.append(test_acc)
            scheduler.step(test_acc)
            if  f1 > best_f1:
                best_f1 = f1
            log(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {test_acc:.4f} | F1: {f1:.4f} | GradNorm: {total_norm:.4f} | Best_F1: {best_f1:.4f}")
    
    return train_losses, test_accs, gradient_norms