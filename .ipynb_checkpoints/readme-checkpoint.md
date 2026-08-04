# Complex State Propagator (CSP)

**State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## TL;DR

We propose **CSP (Complex State Propagator)** , a minimalistic recurrent architecture that **only propagates hidden states** across layers, without output projections at intermediate steps. The state is complex-valued and updated via learned rotations. CSP achieves **100% accuracy** on Parity Check, Mod-3 Counting, and Parenthesis Matching.

**Key insight:** In standard Mamba, `h → y = Ch → next_h = B(y) = B C h`. The two projections can be fused. Why not just propagate `h` directly?

## Architecture

<div align="center">
  <img src="figures/architecture.png" width="600"/>
</div>

### Single Block
<pre>
z_t → Rotate → Recur → (+) → Norm → h_t^(l)
↑                       ↑
└───────── Skip ────────┘
</pre>

- **Rotate**: $\tilde{z}_t = e^{iθ_t} ⊙ z_t$
- **Recur**: $h_t = α_t h_{t-1} + γ_t \tilde{z}_t$
- **Skip**: $\tilde{h}_t = h_t + z_t$
- **Norm**: $h_t^(l) = \tilde{h}_t / |\tilde{h}_t|$

## Results

| Task | Accuracy | F1 | Epochs to 100% |
|------|----------|-----|----------------|
| Parity Check | 100% | 1.0 | ~50 |
| Mod-3 Counting | 100% | 1.0 | ~60 |
| Parenthesis Matching | 100% | 1.0 | ~70 |

### Grokking Phenomenon

We observe clear **grokking** on Mod-3 Counting and Parenthesis Matching:
- Mod-3: remains at 33% for 80 epochs, then jumps to 100% in 10 epochs
- Parenthesis: remains near 50% for 120 epochs, then jumps to 100% in 15 epochs

<div align="center">
  <img src="figures/grokking_curves.png" width="400"/>
  <img src="figures/gradient_norm.png" width="400"/>
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


