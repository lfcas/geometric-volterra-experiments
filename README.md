# Explorations in Geometric Normalization and Continuous Volterra Attention

This repository contains standalone Jupyter notebooks investigating geometric projections and continuous integral formulations across different deep learning modalities: Multi-Layer Perceptrons (MLPs), standard and modern 2D Convolutions (ResNet / ConvNeXt), Graph Neural Networks (GNNs), and Autoregressive Language Models.

The project explores concepts inspired by geometric deep learning and integral equations. Established components—such as Batch Normalization, Layer Normalization, and standard Softmax Attention—are included as reference points (*sanity checks*) to verify that the experimental pipelines, loss functions, and data loaders are behaving as expected under identical optimization conditions.

---

## Execution Environment

All experiments are organized into standalone notebooks that can be executed directly without external setup files:

* **Notebooks `01` to `06`**: Executed in **Kaggle** using default GPU environments (Tesla T4 / P100) with PyTorch.
* **Notebooks `07` and `08`**: Executed in **Google Colab** using an **NVIDIA A100-SXM4-80GB GPU** with JAX/Flax/Optax compiled via XLA under native `bfloat16` precision.

---

## Core Ideas Tested

### 1. Geometric Projections

Instead of using learnable affine parameters (scale and bias), we evaluate two parameter-free projections that keep activation vectors constrained to spherical manifolds:

* **Conical Projection ($\mathbb{S}^{d-1}$)**: Rescales each vector to unit Euclidean length and applies a fixed scaling factor of $\sqrt{d}$ to maintain activation magnitude:
  $$\Pi_{\mathbb{S}^{d-1}}(x) = \sqrt{d} \cdot \frac{x}{\|x\|_2 + \epsilon}$$

* **Equatorial Projection ($\mathbb{S}^{d-2}$)**: Subtracts the mean across features (enforcing zero-sum / zero-trace) before retracting the vector back to the sphere:
  $$\bar{x} = x - \text{mean}(x), \quad \Pi_{\mathbb{S}^{d-2}}(x) = \sqrt{d} \cdot \frac{\bar{x}}{\|\bar{x}\|_2 + \epsilon}$$

### 2. Continuous Volterra-Fredholm Attention

Standard self-attention computes pairwise similarities using an exponential Softmax matrix. Here, we test modeling temporal mixing as a continuous integral equation of the second kind:
$$u(t) = f(t) + \int_0^t \mathcal{K}(t, s) \, v(s) \, ds$$

To implement this causally on discrete sequences:
1. **Chebyshev Polynomial Basis**: Temporal coordinates are represented using orthogonal Chebyshev polynomials $T_m(x)$ evaluated at Gauss-Chebyshev nodes with analytic integration weights $w(s)$.
2. **Dissipative Decay**: A causal exponential decay factor $e^{-\gamma_m(t - s)}$ modulates interactions over time.
3. **Associative Sum**: Because the decay factor decomposes as $e^{-\gamma_m t} \cdot e^{\gamma_m s}$, the operation can be accumulated cumulatively over time:
   $$S_m(t) = \sum_{s=1}^t \left[ \psi_m(s) \, e^{\gamma_m s} \, (k_s \odot v_s) \right]$$
   $$u(t) = q_t \odot \Pi_{\mathbb{S}^{d-1}}\left( \sum_{m=1}^M \lambda_m \, \phi_m(t) \, e^{-\gamma_m t} \, S_m(t) \right)$$
   This formulation evaluates the sequence in $\mathcal{O}(T \cdot M \cdot d)$ without computing a full pairwise exponential attention matrix.

---

## Repository Structure

```text
geometric-deep-learning/
│
├── 01_geometric_mlps.ipynb
│   # Dense Vision (FashionMNIST), Tabular Classification, and Regression (California Housing).
│   # Evaluates Vanilla, BatchNorm, Hyperspherical, and Equatorial projections (5 seeds).
│
├── 02_traditional_convnets.ipynb
│   # Standard 3-stage 2D ConvNets on CIFAR-10 across 5 independent seeds.
│   # Tracks accuracy, loss, stable rank, gradient SNR, and sharpness.
│
├── 03_deep_gnn_cora.ipynb
│   # Multi-depth GCN testing without residual connections (L = 2, 4, 8, 16, 32, 64).
│   # Evaluates node distance, Dirichlet energy, and active node ratios on Cora Planetoid.
│
├── 04_resnets_and_convnext_cosine_head.ipynb
│   # ResNet (3x3) and ConvNeXt topologies with Linear vs. Hyperspherical Cosine Heads.
│   # CIFAR-10 benchmark with data augmentation and cosine annealing (3 seeds).
│
├── 05_canonical_convnext.ipynb
│   # Canonical ConvNeXt (Depthwise 7x7, single norm per block, LayerScale gamma = 1e-6).
│   # Tests 12-epoch multi-seed performance and batch size sensitivity (N from 2 to 256).
│
├── 06_geometric_softmax_transformers.ipynb
│   # Softmax-based Transformers evaluated across 3 training regimes on TinyShakespeare.
│   # Tracks logit magnitude bounds and attention entropy using spherical QK-Norm.
│
├── 07_volterra_autoregressive_lm.ipynb
│   # 51.2M parameter continuous Volterra model in JAX/XLA on Google Colab (A100 GPU).
│   # Evaluates bfloat16 convergence, throughput, and kernel values on WikiText-2.
│
├── 08_volterra_capacity_and_transfer_learning.ipynb
│   # Two-part evaluation in JAX/XLA on Google Colab (A100 GPU):
│   # Part 1: Capacity-matched baseline check on WikiText-2.
│   # Part 2: Pre-training exploration from clean WikiText-103 to WikiText-2.
│
└── README.md
```

---

## Experimental Records

### 1. Multi-Layer Perceptrons (`01_geometric_mlps.ipynb`)

Evaluated across 5 independent seeds (`[42, 1337, 2026, 7, 99]`):
* **FashionMNIST**: 5 layers, hidden dimension 256, 6 epochs.
* **Tabular Classification**: Synthetic dataset (10,000 samples, 32 features), 4 layers, hidden dimension 128, 8 epochs.
* **Continuous Regression**: California Housing (12,000 samples), 4 layers, hidden dimension 128, 12 epochs.

| Architecture | FashionMNIST Acc (%) | Tabular Acc (%) | Regression MSE |
| :--- | :---: | :---: | :---: |
| **Vanilla** *(Reference)* | $87.33\% \pm 0.23\%$ | $96.72\% \pm 0.27\%$ | $0.2189 \pm 0.0052$ |
| **BatchNorm** *(Reference)* | $87.94\% \pm 0.21\%$ | $96.95\% \pm 0.25\%$ | $0.2981 \pm 0.0980$ |
| **Hyperspherical ($\mathbb{S}^{d-1}$)** | $87.99\% \pm 0.15\%$ | $96.69\% \pm 0.18\%$ | $0.2092 \pm 0.0040$ |
| **Equatorial ($\mathbb{S}^{d-2}$)** | $87.00\% \pm 0.35\%$ | $96.01\% \pm 0.53\%$ | $0.3154 \pm 0.0195$ |

#### Latent Stable Rank ($\|H\|_F^2 / \|H\|_2^2$)

| Architecture | SRank Fashion (256D) | SRank Tabular Cls (128D) | SRank Regression (128D) |
| :--- | :---: | :---: | :---: |
| **Vanilla** *(Reference)* | $3.52 \pm 0.43$ | $1.27 \pm 0.02$ | $1.31 \pm 0.03$ |
| **BatchNorm** *(Reference)* | $4.61 \pm 0.19$ | $3.54 \pm 0.06$ | $3.22 \pm 0.36$ |
| **Hyperspherical ($\mathbb{S}^{d-1}$)** | $4.62 \pm 0.30$ | $3.16 \pm 0.09$ | $3.06 \pm 0.04$ |
| **Equatorial ($\mathbb{S}^{d-2}$)** | $4.70 \pm 0.38$ | $14.07 \pm 1.43$ | $3.26 \pm 0.43$ |

---

### 2. Standard 2D Convolutional Networks (`02_traditional_convnets.ipynb`)

Evaluated on CIFAR-10 (45k train, 5k validation, 10k test) using a 3-stage ConvNet ($3 \to 32 \to 64 \to 128$) trained for 8 epochs across 5 independent seeds (`[42, 1337, 2026, 7, 99]`):

| Architecture | Test Accuracy (%) | Test Loss | Stable Rank (128D) | Grad SNR | Sharpness ($\Delta L$) | Time / Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla** *(Reference)* | $66.45\% \pm 1.30\%$ | $0.9533 \pm 0.0310$ | $3.32 \pm 0.26$ | $0.773 \pm 0.277$ | $0.0029 \pm 0.0027$ | $13.1\text{s}$ |
| **BatchNorm** *(Reference)* | $67.57\% \pm 2.14\%$ | $0.9286 \pm 0.0799$ | $4.38 \pm 0.20$ | $0.699 \pm 0.221$ | $0.0383 \pm 0.0280$ | $14.9\text{s}$ |
| **Hyperspherical ($\mathbb{S}^{d-1}$)** | $66.99\% \pm 0.77\%$ | $0.9348 \pm 0.0192$ | $3.12 \pm 0.19$ | $0.777 \pm 0.161$ | $0.0094 \pm 0.0046$ | $21.1\text{s}$ |
| **Equatorial ($\mathbb{S}^{d-2}$)** | $71.98\% \pm 0.92\%$ | $0.8142 \pm 0.0192$ | $5.02 \pm 0.56$ | $0.965 \pm 0.225$ | $0.0106 \pm 0.0070$ | $16.5\text{s}$ |

---

### 3. Deep Graph Convolutional Networks (`03_deep_gnn_cora.ipynb`)

Evaluated on the Cora Planetoid split (140 train, 500 validation, 1,000 test) across depths $L \in [2, 4, 8, 16, 32, 64]$ without residual bypasses across 5 independent seeds (`[42, 1337, 2026, 777, 999]`):

| Architecture | $L=2$ | $L=4$ | $L=8$ | $L=16$ | $L=32$ | $L=64$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla GCN** *(Reference)* | $71.1\% \pm 1.7\%$ | $74.6\% \pm 1.8\%$ | $66.1\% \pm 6.8\%$ | $27.7\% \pm 12.1\%$ | $14.5\% \pm 1.8\%$ | $15.8\% \pm 7.5\%$ |
| **LayerNorm GCN** *(Reference)* | $73.9\% \pm 1.1\%$ | $73.5\% \pm 2.1\%$ | $70.8\% \pm 2.1\%$ | $20.0\% \pm 12.4\%$ | $11.7\% \pm 2.7\%$ | $13.4\% \pm 1.0\%$ |
| **PairNorm GCN** *(Reference)* | $60.5\% \pm 2.5\%$ | $58.2\% \pm 3.7\%$ | $57.2\% \pm 3.2\%$ | $62.4\% \pm 4.3\%$ | $48.0\% \pm 9.4\%$ | $47.4\% \pm 1.4\%$ |
| **Graph-Conical** | $74.1\% \pm 2.3\%$ | $71.1\% \pm 3.0\%$ | $69.6\% \pm 1.1\%$ | $70.5\% \pm 3.1\%$ | $57.5\% \pm 10.8\%$ | $37.5\% \pm 8.5\%$ |
| **Graph-Equatorial** | $73.0\% \pm 2.4\%$ | $71.1\% \pm 1.8\%$ | $70.3\% \pm 1.8\%$ | $69.4\% \pm 2.6\%$ | $68.7\% \pm 3.8\%$ | $47.7\% \pm 8.9\%$ |

#### Structural Metrics at Depth $L=32$

| Architecture | Test Accuracy (%) | Pairwise Distance | Dirichlet Energy | Dead Nodes (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla GCN** *(Reference)* | $14.5\% \pm 1.8\%$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $100.0\%$ |
| **LayerNorm GCN** *(Reference)* | $11.7\% \pm 2.7\%$ | $-0.0000 \pm 0.0000$ | $0.0748 \pm 0.0000$ | $0.0\%$ |
| **PairNorm GCN** *(Reference)* | $48.0\% \pm 9.4\%$ | $0.6553 \pm 0.0421$ | $0.1125 \pm 0.0430$ | $0.0\%$ |
| **Graph-Conical** | $57.5\% \pm 10.8\%$ | $0.2768 \pm 0.0523$ | $0.0059 \pm 0.0015$ | $0.0\%$ |
| **Graph-Equatorial** | $68.7\% \pm 3.8\%$ | $0.7202 \pm 0.0170$ | $0.0123 \pm 0.0050$ | $0.0\%$ |

---

### 4. Residual Networks & Decision Heads (`04_resnets_and_convnext_cosine_head.ipynb`)

Evaluated on CIFAR-10 over 12 epochs with data augmentation and cosine learning rate decay across 3 independent seeds (`[42, 1337, 2026]`):

| Architecture Family | Configuration | Test Accuracy (%) | Test Loss | Stable Rank | Grad SNR | Time / Seed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet Topology** | ResNet-Vanilla *(Reference)* | $83.54\% \pm 0.19\%$ | $0.4907 \pm 0.0057$ | $2.09 \pm 0.07$ | $0.310 \pm 0.008$ | $184.5\text{s}$ |
| | ResNet-BatchNorm *(Reference)* | $86.14\% \pm 0.11\%$ | $0.4125 \pm 0.0012$ | $3.06 \pm 0.07$ | $0.257 \pm 0.023$ | $198.7\text{s}$ |
| | ResNet-Equatorial *(Linear Head)* | $84.25\% \pm 0.21\%$ | $0.4578 \pm 0.0030$ | $3.33 \pm 0.04$ | $0.326 \pm 0.076$ | $276.6\text{s}$ |
| | ResNet-Equatorial *(Cosine Head)* | $84.48\% \pm 0.23\%$ | $0.4580 \pm 0.0102$ | $3.71 \pm 0.08$ | $0.233 \pm 0.013$ | $276.8\text{s}$ |
| | ResNet-Conical *(Cosine Head)* | $75.51\% \pm 0.65\%$ | $0.7104 \pm 0.0168$ | $3.53 \pm 0.04$ | $0.289 \pm 0.046$ | $300.8\text{s}$ |
| **ConvNeXt Topology** | ConvNeXt-LayerNorm *(Reference)* | $76.43\% \pm 0.40\%$ | $0.6785 \pm 0.0078$ | $4.50 \pm 0.11$ | $0.261 \pm 0.007$ | $278.7\text{s}$ |
| | ConvNeXt-Equatorial *(Linear Head)* | $77.11\% \pm 0.44\%$ | $0.6511 \pm 0.0100$ | $4.92 \pm 0.08$ | $0.282 \pm 0.070$ | $253.5\text{s}$ |
| | ConvNeXt-Equatorial *(Cosine Head)* | $76.87\% \pm 0.28\%$ | $0.6612 \pm 0.0070$ | $4.64 \pm 0.06$ | $0.289 \pm 0.018$ | $254.1\text{s}$ |
| | ConvNeXt-Conical *(Linear Head)* | $76.71\% \pm 0.22\%$ | $0.6672 \pm 0.0108$ | $4.73 \pm 0.06$ | $0.263 \pm 0.028$ | $235.2\text{s}$ |
| | ConvNeXt-Conical *(Cosine Head)* | $77.03\% \pm 0.26\%$ | $0.6587 \pm 0.0043$ | $4.40 \pm 0.06$ | $0.284 \pm 0.033$ | $235.5\text{s}$ |

---

### 5. Canonical ConvNeXt & Batch Size Range (`05_canonical_convnext.ipynb`)

#### Multi-Seed Evaluation (12 Epochs, 3 Seeds)

| Architecture | Test Accuracy (%) | Test Loss | Stable Rank | Grad SNR | Sharpness ($\Delta L$) | Time / Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LayerNeXt** *(Reference)* | $76.67\% \pm 0.32\%$ | $0.6699 \pm 0.0124$ | $4.59 \pm 0.09$ | $0.269 \pm 0.015$ | $0.0051 \pm 0.0007$ | $283.3\text{s}$ |
| **EquatorialNeXt** | $77.09\% \pm 0.19\%$ | $0.6600 \pm 0.0032$ | $5.02 \pm 0.11$ | $0.263 \pm 0.013$ | $0.0035 \pm 0.0007$ | $257.4\text{s}$ |
| **ConicalNeXt** | $76.40\% \pm 0.11\%$ | $0.6707 \pm 0.0084$ | $4.83 \pm 0.11$ | $0.264 \pm 0.024$ | $0.0034 \pm 0.0018$ | $239.9\text{s}$ |

#### Batch Size Sensitivity ($N \in [2, 4, 8, 16, 64, 256]$, 1,000 Steps)

| Batch Size ($N$) | BatchNorm (ConvNeXt) | LayerNorm (ConvNeXt) | Conical (ConvNeXt) | Equatorial (ConvNeXt) |
| :---: | :---: | :---: | :---: | :---: |
| $N = 2$ | $20.00\%$ | $25.14\%$ | $24.71\%$ | $25.06\%$ |
| $N = 4$ | $32.86\%$ | $29.84\%$ | $30.51\%$ | $30.35\%$ |
| $N = 8$ | $35.74\%$ | $35.62\%$ | $36.95\%$ | $35.80\%$ |
| $N = 16$ | $45.90\%$ | $43.14\%$ | $44.12\%$ | $43.19\%$ |
| $N = 64$ | $61.92\%$ | $54.71\%$ | $54.38\%$ | $53.37\%$ |
| $N = 256$ | $74.48\%$ | $67.00\%$ | $67.10\%$ | $66.99\%$ |
| **Drop ($N=256 \to 2$)** | **$-54.48\%$** | **$-41.86\%$** | **$-42.39\%$** | **$-41.93\%$** |

---

### 6. Geometric Projections in Softmax Attention (`06_geometric_softmax_transformers.ipynb`)

Evaluated on TinyShakespeare across 3 training setups (3 seeds: `[42, 1337, 2026]`):
* **Regime A**: Shallow ($L=4, \text{LR}=1\times 10^{-3}$, 800 steps).
* **Regime B**: Deeper ($L=8, \text{LR}=1\times 10^{-3}$, 800 steps).
* **Regime C**: Higher Learning Rate ($L=8, \text{LR}=3\times 10^{-3}$, 600 steps).

| Regime | Model Architecture | Validation Loss | Perplexity | Attention Entropy | Peak QK Logit |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Regime A** | LayerNorm-GPT *(Reference)* | $2.3503 \pm 0.0161$ | $10.49 \pm 0.17$ | $2.701 \pm 0.052$ | $42.5 \pm 7.2$ |
| ($L=4, \text{LR}=10^{-3}$) | RMSNorm-GPT *(Reference)* | $2.3690 \pm 0.0239$ | $10.69 \pm 0.26$ | $2.622 \pm 0.057$ | $37.2 \pm 6.3$ |
| | Conical-GPT ($\mathbb{S}^{d-1}$) | $2.3424 \pm 0.0021$ | $10.41 \pm 0.02$ | $2.989 \pm 0.051$ | $5.5 \pm 0.0$ |
| | Equatorial-GPT ($\mathbb{S}^{d-2}$) | $2.3124 \pm 0.0124$ | $10.10 \pm 0.12$ | $2.992 \pm 0.022$ | $5.6 \pm 0.0$ |
| **Regime B** | LayerNorm-GPT *(Reference)* | $2.1908 \pm 0.0092$ | $8.94 \pm 0.08$ | $2.616 \pm 0.076$ | $35.0 \pm 0.9$ |
| ($L=8, \text{LR}=10^{-3}$) | RMSNorm-GPT *(Reference)* | $2.1789 \pm 0.0355$ | $8.84 \pm 0.32$ | $2.607 \pm 0.040$ | $37.2 \pm 3.1$ |
| | Conical-GPT ($\mathbb{S}^{d-1}$) | $2.2007 \pm 0.0165$ | $9.03 \pm 0.15$ | $2.965 \pm 0.031$ | $5.5 \pm 0.0$ |
| | Equatorial-GPT ($\mathbb{S}^{d-2}$) | $2.2293 \pm 0.0370$ | $9.30 \pm 0.35$ | $2.968 \pm 0.010$ | $5.6 \pm 0.0$ |
| **Regime C** | LayerNorm-GPT *(Reference)* | $2.4069 \pm 0.0433$ | $11.11 \pm 0.49$ | $1.770 \pm 0.022$ | $96.7 \pm 4.6$ |
| ($L=8, \text{LR}=3\cdot 10^{-3}$)| RMSNorm-GPT *(Reference)* | $2.4362 \pm 0.0492$ | $11.44 \pm 0.55$ | $1.943 \pm 0.085$ | $82.3 \pm 9.4$ |
| | Conical-GPT ($\mathbb{S}^{d-1}$) | $2.3736 \pm 0.0272$ | $10.74 \pm 0.29$ | $2.900 \pm 0.062$ | $5.6 \pm 0.0$ |
| | Equatorial-GPT ($\mathbb{S}^{d-2}$) | $2.4167 \pm 0.0498$ | $11.22 \pm 0.56$ | $2.858 \pm 0.012$ | $5.6 \pm 0.0$ |

Because the queries and keys are projected onto the unit sphere ($\|q\|_2 = \|k\|_2 = 1$), the maximum dot-product is bounded by $\sqrt{d_k} = \sqrt{32} \approx 5.65$, keeping attention logits bounded across learning rates.

---

### 7. Continuous Volterra Language Model (`07_volterra_autoregressive_lm.ipynb`)

Evaluated on WikiText-2 (GPT-2 BPE tokenizer, $V=50,257$, Context $T=512$, $M=16$ modes, 1,000 steps, batch size 32) using JAX/XLA in `bfloat16` on a **Google Colab NVIDIA A100-SXM4-80GB GPU**:

```text
Model: Volterra-Fundamental | Parameters: 51.2M | Modes: 16 | Precision: bfloat16
XLA Compilation Time: 29.36s
```

| Step | Val Loss | Perplexity (PPL) | Entropy | Kernel Max Value | Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **50** | $5.5463$ | $256.29$ | $4.656$ | $42.5$ | $234.4\text{k tok/s}$ |
| **200** | $4.7771$ | $118.76$ | $4.101$ | $42.2$ | $229.8\text{k tok/s}$ |
| **400** | $4.4586$ | $86.37$ | $3.968$ | $42.4$ | $227.2\text{k tok/s}$ |
| **600** | $4.3095$ | $74.40$ | $3.888$ | $41.7$ | $225.6\text{k tok/s}$ |
| **800** | $4.2421$ | $69.55$ | $3.883$ | $41.4$ | $225.0\text{k tok/s}$ |
| **1000** | $4.2362$ | $69.14$ | $3.889$ | $41.4$ | $224.5\text{k tok/s}$ |

#### Evaluation across 80 Validation Batches

| Architecture | Validation Loss | Final PPL | Mean Entropy | Kernel Max Value | Throughput | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Volterra JAX (M=16)** | $4.2411$ | $69.48$ | $3.889$ | $41.4$ | $224.5\text{k tok/s}$ | $73.0\text{s}$ |

---

### 8. Capacity Testing & Transfer Exploration (`08_volterra_capacity_and_transfer_learning.ipynb`)

Executed on **Google Colab (NVIDIA A100-SXM4-80GB GPU)** in JAX/XLA (`bfloat16`):

#### Part 1: Baseline Check on WikiText-2 (1,200 Steps, $T=512$)

To observe the attention mechanism without the embedding table dominating the count ($V=50,257$), backbone parameters (attention and feed-forward layers) are reported alongside the total:

| Architecture | Backbone Params | Total Params | Tokens / Backbone Param | Best Val Loss | Final PPL | Throughput (A100) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Volterra-Fundamental** ($M=16$) | $3,148\text{k}$ | $16.15\text{M}$ | $0.8:1$ | $4.5035$ | $91.27$ | $510.0\text{k tok/s}$ |
| **Reference Baseline** *(Softmax GPT-2)* | $3,150\text{k}$ | $16.15\text{M}$ | $0.8:1$ | $4.6183$ | $101.61$ | $360.5\text{k tok/s}$ |

#### Part 2: Pre-training Exploration (WikiText-103 $\to$ WikiText-2)

Tested with **Volterra-Micro** ($D=128, L=2, H=4, M=8$, Backbone: $393.6\text{k}$ parameters) pre-trained on clean text from WikiText-103 ($7.74\text{M}$ tokens subset, ratio $19.7:1$) and adapted to WikiText-2 ($2.45\text{M}$ tokens) over 800 steps at $\text{LR} = 3 \times 10^{-4}$:

| Learning Setup | Pre-train Source | Adaptation Steps | WikiText-2 Val Loss | Final WikiText-2 PPL |
| :--- | :--- | :---: | :---: | :---: |
| **Trained from Scratch** | None *(WikiText-2 only)* | $800$ | $5.5597$ | $259.76$ |
| **Pre-trained + Fine-tuned** | WikiText-103 *(Clean Text)* | $800$ | $4.7308$ | $113.39$ |

Pre-training on the broader text corpus before fine-tuning yields a perplexity of $113.39$ compared to $259.76$ when starting from random initialization under the same 800-step budget.

---

## Known Constraints & Scope

1. **Modal Truncation Horizon**: While the Volterra operator scales linearly as $\mathcal{O}(T \cdot M \cdot d)$, the current implementation discretizes the Chebyshev polynomial base over a fixed context length ($T = 512$). Dynamic sequence length extrapolation requires runtime polynomial re-interpolation.
2. **Low-Dimensional Modal Rank**: In latent spaces where $d \le 32$, setting $M \ge d$ introduces linear dependence in the associative state $S_m(t)$, capping the performance benefit of higher modal resolutions.

---

## Citation & Reference

```bibtex
@article{volterra_geometric_dl_2026,
  title={Continuous Volterra-Fredholm Integral Operators and Hyperspherical Geometric Constraints in Deep Neural Architectures},
  author={Anonymous},
  year={2026},
  journal={Repository of Geometric Deep Learning}
}
```