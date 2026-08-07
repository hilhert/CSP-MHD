import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, test_accs, save_path='.', file_name="training_curves.png", show=True):
    """
    绘制训练曲线：Loss + Accuracy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    ax1.plot(train_losses, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    # test_accs 每10轮记录一次，需要对齐 x 轴
    epochs = range(0, len(train_losses), 10)
    if len(epochs) > len(test_accs):
        epochs = epochs[:len(test_accs)]
    ax2.plot(epochs, test_accs, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")
    

def plot_training_curves_f1(train_losses, f1s, save_path='.', file_name="training_curves_f1.png", show=True):
    """
    绘制训练曲线：Loss + Accuracy
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 曲线
    ax1.plot(train_losses, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    # test_accs 每10轮记录一次，需要对齐 x 轴
    epochs = range(0, len(train_losses), 10)
    if len(epochs) > len(f1s):
        epochs = epochs[:len(f1s)]
    ax2.plot(epochs, f1s, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy(F1 hit rate)')
    ax2.set_title('Test Accuracy(F1 hit rate)')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")    
    
    
    


def plot_grokking_analysis(test_accs, gradient_norms, train_losses=None, 
                          save_path='.', file_name='grokking_analysis.png', show=True):
    """
    绘制 Grokking 分析图：准确率 + 梯度范数
    
    Args:
        test_accs: 每10轮记录的测试准确率列表
        gradient_norms: 每轮的梯度范数列表
        train_losses: 可选，每轮的训练损失
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：准确率曲线（标注 grokking 位置）
    epochs_acc = range(0, len(test_accs) * 10, 10)
    if len(epochs_acc) > len(test_accs):
        epochs_acc = epochs_acc[:len(test_accs)]
    
    ax1.plot(epochs_acc, test_accs, 'b-', linewidth=2, label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Grokking: Test Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random guess')
    
    # 找准确率跳变点（从 < 0.9 到 > 0.9）
    grokking_epoch = None
    for i, acc in enumerate(test_accs):
        if acc > 0.9:
            grokking_epoch = i * 10
            break
    if grokking_epoch is not None:
        ax1.axvline(x=grokking_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Grokking at epoch {grokking_epoch}')
    ax1.legend()
    
    # 右图：梯度范数曲线
    ax2.plot(gradient_norms, 'g-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm Dynamics')
    ax2.grid(True, alpha=0.3)
    
    # 标出梯度尖峰位置
    if gradient_norms:
        peak_idx = np.argmax(gradient_norms)
        ax2.axvline(x=peak_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Gradient spike at epoch {peak_idx}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_gradient_norm(gradient_norms, save_path='.',file_path='gradient_norm.png', show=True):
    """
    单独绘制梯度范数曲线（论文 Figure 4 用）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    plt.plot(gradient_norms, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm During Training')
    plt.grid(True, alpha=0.3)
    
    if gradient_norms:
        peak_idx = np.argmax(gradient_norms)
        plt.axvline(x=peak_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Spike at epoch {peak_idx}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,file_name), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {save_path}")
    
def plot_grokking_analysis_f1(test_f1s, gradient_norms, train_losses=None, 
                          save_path='.', file_name='grokking_analysis_f1.png', show=True):
    """
    绘制 Grokking 分析图：F1 + 梯度范数
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：F1曲线
    epochs_f1 = range(0, len(test_f1s) * 10, 10)
    if len(epochs_f1) > len(test_f1s):
        epochs_f1 = epochs_f1[:len(test_f1s)]
    
    ax1.plot(epochs_f1, test_f1s, 'b-', linewidth=2, label='F1 Score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Grokking: F1 Score')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.001, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    grokking_epoch = None
    for i, f1 in enumerate(test_f1s):
        if f1 >= 0.9:
            grokking_epoch = i * 10
            break
    if grokking_epoch is not None:
        ax1.axvline(x=grokking_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Grokking at epoch {grokking_epoch}')
    ax1.legend()
    
    # 右图：梯度范数曲线
    ax2.plot(gradient_norms, 'g-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm Dynamics')
    ax2.grid(True, alpha=0.3)
    
    if gradient_norms:
        peak_idx = int(np.argmax(gradient_norms))
        ax2.axvline(x=peak_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Gradient spike at epoch {peak_idx}')
    ax2.legend()
    
    plt.tight_layout()
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Figure saved: {full_path}")