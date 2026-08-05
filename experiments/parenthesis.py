import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from csp import CSP, generate_parenthesis_data, create_dataloaders
from utils import (
    train_model, evaluate_model, evaluate_model_f1, focal_loss,
    setup_logging, plot_training_curves, plot_grokking_analysis
)


def main():
    log_file = setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X, y = generate_parenthesis_data(10000, 16)
    train_loader, test_loader = create_dataloaders(X, y, batch_size=64)
    
    model = CSP(hidden_dim=128, output_dim=2, num_layers=3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    losses, accs,grad_norms = train_model(
        model, train_loader, test_loader,
        epochs=300, lr=0.0001, device=device,loss_fn=focal_loss
    )
    
    # ★★★ 绘图 ★★★
    plot_training_curves(losses, accs)
    plot_grokking_analysis(accs, grad_norms, losses)
    
    print(f"Final Acc: {evaluate_model(model, test_loader, device):.4f}")
    print(f"Final F1: {evaluate_model_f1(model, test_loader, device):.4f}")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    main()