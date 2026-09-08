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
    
    # training params
    trn_ba_sz  =128
    tst_ba_sz  = trn_ba_sz//2
    v_train    = 256000
    v_test     = v_train//10
    n          =9    #modular number
    max_digits = 1
    max_val    =9
    min_val    =0
    max_terms  =3 
    epochs = 50
   
    
    experiment,data_mode = 'symseq','complete' # or atten_mhead , atten, 
    # load model config file 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, experiment)
    
    os.makedirs(model_path, exist_ok=True)
    
    model_config_file  =  os.path.join(model_path,"config.json") 
    vocab_config_file  =  os.path.join(model_path,"vocab.json")
    try:
        with open(model_config_file,"r") as f:
            model_config = dict(**json.load(f))
        
    except:
        print("Model config file does not exist, please prepare one at model path named config.json with valid json format!")
    
    #prepare datasets
    # Airthmetic Data Set, n reperesents the modular number
    gen = ModNArithmeticGenerator(n=n,simple = False)
    train_dataset = SymbolicArithmeticDataset(
        v_train, max_terms=max_terms, max_digits=max_digits, min_val=min_val, max_val=max_val,
        generate_expression_func=gen, vocab=None, mode=data_mode
    )
    test_dataset = SymbolicArithmeticDataset(
        v_test, max_terms=max_terms, max_digits=max_digits, min_val=min_val, max_val=max_val,
        generate_expression_func=gen,vocab=None ,mode=data_mode
    )
    
    
    
    
    if not os.path.exists(vocab_config_file):
        vocab_config = train_dataset.char2idx
        vocab_config["vocab_size"] = len(train_dataset.char2idx)
        with open(vocab_config_file,"w") as f:     
            json.dump(train_dataset.char2idx, f ,default=lambda o: o.__dict__, indent=4 )
            
    else:
        with open(vocab_config_file,"r") as f:
            vocab_config = dict(**json.load(f))
        
   
    train_loader = DataLoader(train_dataset, batch_size=trn_ba_sz, shuffle=True,drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=tst_ba_sz, shuffle=False,drop_last=True)
    
    
    
    fig_path = os.path.join(model_path, "./figure")
    os.makedirs(fig_path, exist_ok=True)

    
    exp_file = "{}_{}_{}.json".format(experiment,model_config["global"]["identify"],data_mode)
    exp_path = os.path.join(fig_path,exp_file)
        
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            experiment_results = dict(**json.load(f))
       
    
    model_name = "{}_{}_{}.pt".format(experiment,model_config["global"]["identify"],data_mode)
    cp_path = os.path.join(model_path,model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
 
    
    model = CSP_Seq2Seq( model_config, vocab_config).to(device)
    
    
    # Load model if exists! 
    checkpoint=None
    if os.path.exists(cp_path):
        print(f"Loading existence model: {cp_path}")
        checkpoint = torch.load(cp_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        start_epoch = checkpoint.get('epoch', 0)
        with torch.no_grad():
            model.embedding.weight[vocab_config["<SOS>"]].zero_()
            model.embedding.weight[vocab_config["<SOS>"]].requires_grad = False
      
    else:
        print("New Model Created")
       
        start_epoch = 0
    if os.path.exists(exp_path):
        
        with open(exp_path) as f:
            experiment_results = dict(**json.load(f))
    else:
        experiment_results = {"losses":[],"grad_norms":[],"accs":[]}
    
   
    print(f"numel of params: {sum(p.numel() for p in model.parameters()):,}")
    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=5e-4)
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state restored (including adjusted learning rate).")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_config["<PAD>"],reduction='none')

   
    
    best_accu = 0.0
   
    
    for epoch in range(start_epoch,start_epoch+epochs):
        loss,grad_norm = train_model_seq(model, train_loader, optimizer, criterion, device=device,focal=None)
        acc = evaluate_seq(model, test_loader, device, vocab_config["<PAD>"], vocab_config["<EOS>"],debug=True)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
        experiment_results["losses"].append(loss)
        experiment_results["grad_norms"].append(grad_norm)
        experiment_results["accs"].append(acc)
        
        scheduler.step()
        if (epoch+1)%10==0:
            
            train_dataset.update(v_train, max_terms=max_terms, max_digits=max_digits, min_val=min_val, max_val=max_val)
            test_dataset.update(v_test, max_terms=max_terms, max_digits=max_digits, min_val=min_val, max_val=max_val)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Rebuild dataloader!
            train_loader = DataLoader(train_dataset, batch_size=trn_ba_sz, shuffle=True,drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=tst_ba_sz, shuffle=False,drop_last=True)
            
        
        if best_accu<acc:
            save_checkpoint(model, optimizer, epoch, loss, acc, cp_path)
            best_accu=acc
    
        # save train, test data 
        with open(exp_path,'w') as f:
            json.dump(experiment_results, f ,default=lambda o: o.__dict__, indent=4 )
            #f.write(ser_exp)
    

if __name__ == "__main__":
    main()