import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from csp import CSP_Seq2Seq, ModNArithmeticGenerator, SymbolicArithmeticDataset
from utils import (
    train_model_seq,evaluate_seq, focal_loss,
    setup_logging, plot_training_curves_f1, plot_grokking_analysis_f1, save_checkpoint, load_checkpoint
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'symseq')
fig_path = os.path.join(model_path, 'figure')
os.makedirs(model_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)
model_name = "symseq.pt"
cp_path = os.path.join(model_path,model_name)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 数据集
    gen = ModNArithmeticGenerator(n=3)
    mode = 'complete'
    train_dataset = SymbolicArithmeticDataset(
        20000, max_terms=3, max_digits=1, min_val=0, max_val=2,
        generate_expression_func=gen, vocab=None, mode=mode
    )
    test_dataset = SymbolicArithmeticDataset(
        2000, max_terms=3, max_digits=1, min_val=0, max_val=2,
        generate_expression_func=gen,vocab=None ,mode=mode
    )
    '''
    print(f"char2idx: {test_dataset.char2idx}")
    
    for i,item in enumerate(test_dataset):
        print(item)
        break
    
    return
    '''
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 获取特殊 token 索引
    pad_idx = train_dataset.char2idx['<PAD>']
    eos_idx = train_dataset.char2idx['<EOS>']

    # 模型
    vocab_size = train_dataset.vocab_size
    model = CSP_Seq2Seq(vocab_size, hidden_dim=64, num_layers=3).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 训练循环
    epochs = 50
    for epoch in range(epochs):
        loss = train_model_seq(model, train_loader, optimizer, criterion, device, pad_idx)
        acc = evaluate_seq(model, test_loader, device, pad_idx, eos_idx,debug=True)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'symseq/csp_symbolic.pt')
    print("模型已保存为 csp_symbolic.pt")
'''
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
'''

if __name__ == "__main__":
    main()