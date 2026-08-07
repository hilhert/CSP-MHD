# Complex State Propagator (CSP)

**State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking**

[![arXiv](https://img.shields.io/badge/arXiv-2608.03425-b31b1b.svg)](https://arxiv.org/abs/2608.03425)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## TL;DR

We propose **CSP (Complex State Propagator)** , a minimalistic recurrent architecture that **only propagates hidden states** across layers, without output projections at intermediate steps. The state is complex-valued and updated via learned rotations. CSP achieves **100% accuracy** on Parity Check, Mod-3 Counting, and Parenthesis Matching.

**Key insight:** In standard Mamba, `h → y = Ch → next_h = B(y) = B C h`. The two projections can be fused. Why not just propagate `h` directly?


If you find this work useful, please cite it in your paper:

```bibtex
@article{li2026state,
  title={State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking},
  author={Li, Xiaohe and Lu, Yang},
  journal={arXiv preprint arXiv:2608.03425},
  year={2026}
}
```



## Architecture

CSP is a minimal recurrent architecture that **only propagates hidden states** across layers, without output projections at intermediate steps.

### Overall Flow

```
Input x (T steps)
    │
    ▼
  Embedding
    │
    ▼
  CSP Block 1 ──► h^(1)
    │
    ▼
  CSP Block 2 ──► h^(2)
    │
    ▼
    ...
    │
    ▼
  CSP Block L ──► h^(L)
    │
    ▼
  Phase Decoder (atan2->[cos,sin])
    │
    ▼
  Output y_hat
```

### Single CSP Block

Each block has four stages:

```
z_t ──► Rotate ──► Recur ──┐
     │                     │    
     └───── Skip ──────────┼──► (+) ──► Norm ──► h_t
```

- **Rotate**: $\tilde{z}_t = e^{iθ_t} ⊙ z_t$
- **Recur**: $h_t = α_t h_{t-1} + γ_t \tilde{z}_t$
- **Skip**: $\tilde{h}_t = h_t + z_t$
- **Norm**: $h_t^(l) = \tilde{h}_t / |\tilde{h}_t|$


### Key Principles

1. **State-Only Propagation** – No output projections between layers.
2. **Complex Rotation** – Per-dimension independent rotation.
3. **Phase Decoding** – Final prediction from `atan2(Im(h), Re(h))`.
4. **Block-Level SiLU** – Nonlinearity only at block boundaries.



## Results

| Task | Accuracy | F1 | Epochs to 100% |
|------|----------|-----|----------------|
| Parity Check | 100% | 1.0 | ~20 |
| Mod-3 Counting | 100% | 1.0 | ~60 |
| Parenthesis Matching | 100% | 1.0 | ~10 |

### Grokking Phenomenon

We observe clear **grokking** on Parity, Mod-3 Counting and Parenthesis Matching:
- parity: 

<div align="center">
  <img src="./experiments/parity/figure/grokking_analysis.png" width="400"/>
  <img src="./experiments/parity/figure/training_curves.png" width="400"/>
</div>

- Mod-3: 

<div align="center">
  <img src="./experiments/mod3/figure/grokking_analysis.png" width="400"/>
  <img src="./experiments/mod3/figure/training_curves.png" width="400"/>
</div>

- Parenthesis: using f1 score to demostrate grokking, since accuray lost meaning in this problem

<div align="center">
  <img src="./experiments/parenthesis/figure/grokking_analysis_f1.png" width="400"/>
  <img src="./experiments/parenthesis/figure/training_curves_f1.png" width="400"/>
</div>


## Quick Start

```bash
# Clone
git clone https://github.com/hilhert1987/CSP.git
cd CSP

# Install
pip install -r requirements.txt
pip install -e .

# Run experiments
python experiments/parity.py
python experiments/mod3.py
python experiments/parenthesis.py

# Jupyter demo
jupyter notebook notebooks/demo.ipynb

```

## Requirements

- torch>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- tqdm>=4.65.0
- scikit-learn>=1.3.0
- safetensors >=0.8.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Xiaohe Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.