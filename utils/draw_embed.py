import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def visualize_embeddings(checkpoint_path, save_dir='./figures', model_name='MHA-CSP'):
    """
    extract embedding from checkpoint and visualize
    
    Args:
        checkpoint_path: path of model weights (.pt)
        save_dir: location to save figure
        model_name: model name
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # process two possible format ： direct state_dict or dict containing model_state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 2. extract embedding
    embed_weight = None
    for key in state_dict.keys():
        if 'embedding.weight' in key:
            embed_weight = state_dict[key].numpy()
            print(f"Found embedding: {key}, shape={embed_weight.shape}")
            break
        elif 'embedding' in key and 'weight' in key:
            embed_weight = state_dict[key].numpy()
            print(f"Found embedding: {key}, shape={embed_weight.shape}")
            break
    
    if embed_weight is None:
        raise ValueError("No embedding layer found in state_dict! Available keys:\n" + "\n".join(state_dict.keys()))
    
    vocab_size, embed_dim = embed_weight.shape
    
    # 3. extract digit 0-9's embedding（suppose digit token在0-9位置）
   
    digit_embeddings = embed_weight[0:10]
    
    
    print(f"Digit embeddings shape: {digit_embeddings.shape}")
    
    # 4. projct to 2D
    print("Performing PCA (16 components)...")
    pca = PCA(n_components=min(8, embed_dim))
    reduced = pca.fit_transform(digit_embeddings)
    
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(5, len(digit_embeddings)-1), 
                random_state=42, init='pca', learning_rate='auto')
    embedded_2d = tsne.fit_transform(reduced)
    
    # 5. visualize - main
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    
    for i in range(10):
        x, y = embedded_2d[i, 0], embedded_2d[i, 1]
        ax.scatter(x, y, color=colors[i], s=300, edgecolors='black', linewidth=2, zorder=3)
        ax.annotate(str(i), (x, y), fontsize=18, ha='center', va='center', 
                   weight='bold', color='white', zorder=4)
    
    # plot lines（with digit orders）
    for i in range(9):
        ax.plot([embedded_2d[i, 0], embedded_2d[i+1, 0]], 
                [embedded_2d[i, 1], embedded_2d[i+1, 1]], 
                'gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title(f'{model_name}: Embedding Visualization of Digits 0-9\n(48 epochs, {vocab_size} vocab, {embed_dim} dim)', 
                 fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # add labels
    ax.text(0.05, 0.95, 'Circular structure indicates\nmod-9 periodic embedding', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_embedding_viz.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")
    
    # 6. compute the scores
    # compute similarity of nearby digits
    angles = np.arctan2(embedded_2d[:, 1], embedded_2d[:, 0])
    sorted_indices = np.argsort(angles)
    print(f"Angle order (should be close to [0,1,2,3,4,5,6,7,8,9]): {sorted_indices}")
    
   
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(embedded_2d))
    ring_quality = 1.0 / (1.0 + np.std([dist_matrix[i, (i+1)%10] for i in range(10)]))
    print(f"Ring quality score (higher = better circle): {ring_quality:.4f}")
    
    # 7. PCA  main component contribution scores!
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.bar(range(1, len(cumsum)+1), pca.explained_variance_ratio_, alpha=0.6, label='Individual')
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'r-', linewidth=2, label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_path = os.path.join(save_dir, f'{model_name}_pca_variance.png')
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {pca_path}")
    
    return embedded_2d, sorted_indices, ring_quality


def compare_embeddings(checkpoint_paths, model_names, save_dir='./figures'):
    """
    compare embedding distribution with different models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(checkpoint_paths), figsize=(5*len(checkpoint_paths), 5))
    if len(checkpoint_paths) == 1:
        axes = [axes]
    
    for idx, (path, name) in enumerate(zip(checkpoint_paths, model_names)):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        embed_weight = None
        for key in state_dict.keys():
            if 'embedding.weight' in key:
                embed_weight = state_dict[key].numpy()
                break
        
        if embed_weight is None:
            print(f"Warning: No embedding in {path}")
            continue
        
        digit_emb = embed_weight[0:10]
        
        # t-SNE
        pca = PCA(n_components=min(50, digit_emb.shape[1]))
        reduced = pca.fit_transform(digit_emb)
        tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(reduced)
        
        ax = axes[idx]
        colors = plt.cm.rainbow(np.linspace(0, 1, 10))
        for i in range(10):
            x, y = emb_2d[i, 0], emb_2d[i, 1]
            ax.scatter(x, y, color=colors[i], s=200, edgecolors='black', linewidth=2, zorder=3)
            ax.annotate(str(i), (x, y), fontsize=14, ha='center', va='center', weight='bold')
        
        # linking
        for i in range(9):
            ax.plot([emb_2d[i, 0], emb_2d[i+1, 0]], 
                    [emb_2d[i, 1], emb_2d[i+1, 1]], 
                    'gray', linestyle='--', alpha=0.3)
        
        ax.set_title(f'{name}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'embedding_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


# ============= usage =============
if __name__ == "__main__":
   
    checkpoint_path = "symseq/atten_rbf_complete.pt"  # switch to your rail path!
    save_dir = "./figures"
    
    # plot embedding
    visualize_embeddings(checkpoint_path, save_dir, model_name='MHA-CSP')
    
    # if you want to compare different checkpoints：
    # compare_embeddings(
    #     checkpoint_paths=["csp_relax.pt", "atten_rbf_complete.pt"],
    #     model_names=["CSP (Mamba)", "MHA-CSP"],
    #     save_dir=save_dir
    # )