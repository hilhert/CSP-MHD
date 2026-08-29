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
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from csp import CSP_Seq2Seq, ModNArithmeticGenerator, SymbolicArithmeticDataset
from utils import (
    train_model_seq,evaluate_seq, focal_loss,
    setup_logging, plot_training_curves_f1, plot_grokking_analysis_f1, save_checkpoint, load_checkpoint
)



def main():
    experiment,data_mode,model_mode = 'symseq','complete',"atten_rbf" # or atten_mhead , atten, 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, experiment)
    fig_path = os.path.join(model_path, 'figure')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    log_file = setup_logging()
    exp_file = "{}_{}_{}.json".format(experiment,model_mode,data_mode)
    exp_path = os.path.join(fig_path,exp_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Airthmetic Data Set, n reperesents the modular number
    gen = ModNArithmeticGenerator(n=9,simple = False)
    model_name = "{}_{}_{}.pt".format(experiment,model_mode,data_mode)
    cp_path = os.path.join(model_path,model_name)
    train_dataset = SymbolicArithmeticDataset(
        200000, max_terms=3, max_digits=1, min_val=0, max_val=9,
        generate_expression_func=gen, vocab=None, mode=data_mode
    )
    test_dataset = SymbolicArithmeticDataset(
        20000, max_terms=3, max_digits=1, min_val=0, max_val=9,
        generate_expression_func=gen,vocab=None ,mode=data_mode
    )
    
    '''
    print(f"char2idx: {train_dataset.char2idx}")
    
    for i,item in enumerate(train_dataset):
        print(item)
        break
    
    return
    '''
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # get special token index
    pad_idx = train_dataset.char2idx['<PAD>']
    eos_idx = train_dataset.char2idx['<EOS>']
    sos_idx = train_dataset.char2idx['<SOS>']
    print(f" sos_idx:{sos_idx}\n pad_idx:{pad_idx}\n eos_idx:{eos_idx} ")
    

    # model param
    vocab_size = train_dataset.vocab_size
    hidden_dim = 256
    n_head = 8
    num_layers = 16
    embed_dim  = 64
    model = CSP_Seq2Seq(vocab_size, head_dim = hidden_dim//n_head,n_head=n_head, num_layers=num_layers,embed_dim=embed_dim,sos_idx=sos_idx,pad_idx=pad_idx,eos_idx=eos_idx,model_mode=model_mode).to(device)
    if os.path.exists(cp_path):
        print(f"Loading existence model: {cp_path}")
        checkpoint = torch.load(cp_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        start_epoch = checkpoint.get('epoch', 0)
        with torch.no_grad():
            model.embedding.weight[sos_idx].zero_()
            model.embedding.weight[sos_idx].requires_grad = False
      
    else:
        print("New Model Created")
       
        start_epoch = 0
    if os.path.exists(exp_path):
        
        with open(exp_path) as f:
            experiment_results = dict(**json.load(f))
    else:
        experiment_results = {"losses":[],"grad_norms":[],"accs":[]}
    
    #model = CSP_Seq2Seq(vocab_size, hidden_dim=64, num_layers=7).to(device)
    print(f"numel of params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*1.5, eta_min=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,reduction='none')

    # traning epoches
    epochs = 50
    
    best_accu = 0.0
    #focal = {"alpha":0.25, "gamma":2.0}
    
    for epoch in range(start_epoch,start_epoch+epochs):
        loss,grad_norm = train_model_seq(model, train_loader, optimizer, criterion, device=device,focal=None)
        acc = evaluate_seq(model, test_loader, device, pad_idx, eos_idx,debug=True)
        #print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
        #best_acc = 0
        '''
        if epoch % 10 == 0:
            test_accs.append(acc)
            #scheduler.step(test_acc)
            if  acc > best_acc:
                save_checkpoint(model, optimizer, epoch, loss, acc, cp_path)
            #log(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f})
        '''
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
        experiment_results["losses"].append(loss)
        experiment_results["grad_norms"].append(grad_norm)
        experiment_results["accs"].append(acc)
        
        scheduler.step()
        if (epoch+1)%10==0:
            
            train_dataset.update(200000, max_terms=3, max_digits=1, min_val=0, max_val=9)
            test_dataset.update(20000, max_terms=3, max_digits=1, min_val=0, max_val=9)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Rebuild dataloader!
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
        
        if best_accu<acc:
            save_checkpoint(model, optimizer, epoch, loss, acc, cp_path)
            best_accu=acc
    
        # save train, test data 
        with open(exp_path,'w') as f:
            json.dump(experiment_results, f ,default=lambda o: o.__dict__, indent=4 )
            #f.write(ser_exp)
  

if __name__ == "__main__":
    main()