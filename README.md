# MHA-CSP: Mahalanobis-Based Multi-Head Attention for Complex State Propagation

[![arXiv](https://img.shields.io/badge/arXiv-2608.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2608.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## TL;DR

We propose **MHA-CSP**, extending CSP with a **distance-based multi-head attention mechanism** inside each propagation block. Instead of Q/K/V projections, MHA-CSP constructs attention directly from **Mahalanobis distances** with tree-structured accumulation and cross-head meshing.

With **119K parameters** and **teacher forcing at the final hidden state**, MHA-CSP achieves **50% accuracy** on nested arithmetic expressions.

```bibtex
@article{li2026mhacsp,
  title={Mahalanobis-Based Multi-Head Attention for Complex State Propagation},
  author={Li, Xiaohe},
  journal={arXiv preprint arXiv:2608.XXXXX},
  year={2026}
}
```

## Architecture

MHA-CSP stacks L identical blocks. Each block:

1. **Complex Propagation**: $h_t = h_{t-1} \cdot e^{i\theta_t}$, $\theta_t = f_\theta(x_t)$
2. **Multi-Head Split**: $\{h^{(1)}, h^{(2)}, ..., h^{(H)}\}$
3. **Per-Head Mahalanobis Distance**: $D^{(k)}_{ij} = \|h_i - h_j\|_{M^{(k)}}$
4. **Tree Accumulation**: $C^{(k)}_{ij} = \text{accu}(j) - \text{accu}(i)$, $\text{accu}(j) = \sum_{t=1}^j \exp(e_t)$
5. **LogSumExp Correction**: $\tilde{C}_{ij} = \log(1 + C_{ij})$
6. **Global Summary**: $c_k = \tilde{C}^{(k)}_{T,1}$
7. **Cross-Head Meshing**: $A_{\text{head}} = \text{softmax}(c^\top B c)$
8. **Fused Distance**: $D_{\text{fused}} = \sum_k A_{\text{head}}[:,k] \cdot D^{(k)}$
9. **Attention**: $\text{Attn}(i,j) = \exp(-D_{\text{fused},ij})$
10. **State Update**: Weighted sum over sequence

Finally, phase decoder outputs the prediction.

## Key Principles

- **No Q/K/V** – attention from distances
- **Mahalanobis** – learnable positive definite metric
- **Tree Accumulation** – hierarchical structure via LogSumExp
- **Head Meshing** – dynamic cross-head collaboration via $c^\top B c$
- **Wirtinger Isometry** – gradient-preserving complex propagation
- **119K parameters** – efficient structured reasoning

## Results

| Task | MHA-CSP | Vanilla CSP | ARFormer | GDN | LSTM/GRU |
|------|---------|-------------|----------|-----|----------|
| Parenthesis Matching | None | 100% | 38.7% | 22.1% | 12-16% |
| Arithmetic + Repeat | **50.3%** | 52.1% | 32.1% | 18.3% | 8-11% |
| Parity Check | None | 100% | 91.2% | 78.4% | 62-65% |

### The "Repeat" Trick

Target format: `Answer + "repeat" + Input + Answer`

Example: `8 repeat ((2 + (0 - 3)) + ((0 - 3) + 2) + (2 - 1)) mod 9 = 8`

This alone improves accuracy from 34.7% to 50.3%.

### Grokking & Visualization

- Grokking: accuracy stays near random for 15-20 epochs, then jumps to 50%
- Embeddings: digits 0-9 form a circle in complex space, showing learned modulo-9 structure

## Quick Start

```bash
git clone https://github.com/hilhert/CSP-MHD.git
cd CSP-MHD
pip install -r requirements.txt
pip install -e .
python sim_sequence.py          # main experiment
python train_mini_arformer.py   # baseline
```

## Requirements

- torch>=2.0.0, numpy>=1.24.0, matplotlib>=3.7.0, tqdm>=4.65.0, scikit-learn>=1.3.0, safetensors>=0.8.0

## Citation

```bibtex
@article{li2026mhacsp,
  title={Mahalanobis-Based Multi-Head Attention for Complex State Propagation},
  author={Li, Xiaohe},
  journal={arXiv preprint arXiv:2608.XXXXX},
  year={2026}
}
```

## License

MIT License. Copyright (c) 2026 Xiaohe Li
