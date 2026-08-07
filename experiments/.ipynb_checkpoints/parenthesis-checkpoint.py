import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from csp import CSP, generate_parenthesis_data, create_dataloaders
from utils import (
    train_model, evaluate_model, evaluate_model_f1, focal_loss,
    setup_logging, plot_training_curves_f1, plot_grokking_analysis_f1, save_checkpoint, load_checkpoint
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'parenthesis')
fig_path = os.path.join(model_path, 'figure')
os.makedirs(model_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)
model_name = "parenthesis.pt"
cp_path = os.path.join(model_path,model_name)

def main():
    
    log_file = setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X, y = generate_parenthesis_data(10000, 16)
    train_loader, test_loader = create_dataloaders(X, y, batch_size=64)
    
    model = CSP(hidden_dim=32, output_dim=2, num_layers=3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    losses, f1s,accs,grad_norms = train_model(
        model, train_loader, test_loader,weight_decay=1e-4,
        epochs=300, lr=0.0001, device=device,loss_fn=focal_loss, cp_path=cp_path
    )
    
    # ★★★ 绘图 ★★★
    plot_training_curves_f1(losses, f1s,save_path=fig_path)
    plot_grokking_analysis_f1(f1s, grad_norms, losses,save_path=fig_path)
    #plot_gradient_norm()
    
    print(f"Final Acc: {evaluate_model(model, test_loader, device):.4f}")
    print(f"Final F1: {evaluate_model_f1(model, test_loader, device):.4f}")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    main()