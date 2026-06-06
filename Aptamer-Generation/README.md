# 🧬 AI-Driven Aptamer Generation via WGAN-GP

## Overview

This repository contains the generative component of the aptamer discovery pipeline. The goal is to produce novel, chemically valid DNA/RNA aptamer sequences using a modified Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP).

The architecture is inspired by [RNAGen](https://github.com/) but has been significantly adapted for the specific task of aptamer generation, including:
- **Gumbel-Softmax** trick for differentiable sampling from discrete nucleotide distributions.
- **K-mer matching loss** to enforce biological motif similarity.
- **Entropy regularization** to prevent mode collapse and ensure sequence diversity.
- **Upsample + Conv1d** blocks to eliminate checkerboard artifacts common in transposed convolutions.

## Repository Structure

```
aptamer-generation/
├── data/
│   ├── train_rna_dataset.csv      # Training dataset for RNA aptamers
│   └── train_dna_dataset.csv      # Training dataset for DNA aptamers
├── weights/
│   ├── generator_rna.pth          # RNA generator weights
│   ├── critic_rna.pth             # RNA critic weights
│   ├── generator_dna.pth          # DNA generator weights
│   └── critic_dna.pth             # DNA critic weights
├── training_results/
│   ├── rna_generation_results.txt # Training logs and metrics for RNA
│   └── dna_generation_results.txt # Training logs and metrics for DNA
├── seq_gen.py                     # Main training script (RNA)
├── seq_gen_dna.py                 # Main training script (DNA)
├── metrics.py                     # Evaluation metrics (GC, MFE, Levenshtein)
└── README.md
```

## Architecture

### Generator
The generator maps a latent noise vector (z ∈ ℝ^64) to a sequence of length 80 over the alphabet {A, C, G, U/T, X}, where X is a padding token.

1. **Linear Projection:** z → ℝ^{128 × 20}
2. **Upsampling Blocks:** Two stages of Upsample(scale=2) → Conv1d to increase sequence resolution (20 → 40 → 80).
3. **Residual Blocks:** Two ResBlock modules with BatchNorm1d for stable gradient flow.
4. **Output Layer:** Conv1d(32, 5, 1) followed by Gumbel-Softmax with annealed temperature τ.

### Critic (Discriminator)
The critic estimates the Wasserstein distance between real and generated distributions.

1. **Embedding:** Conv1d(5, 64, 5)
2. **Downsampling:** Three Conv1d layers with stride=2 and LeakyReLU(0.2).
3. **Classification:** Linear(256 × 20, 1)

### Loss Functions

#### Critic Loss
```
L_D = E[D(fake)] - E[D(real)] + λ · GP(real, fake)
```
where λ = 10 and GP is the gradient penalty term enforcing the Lipschitz constraint.

#### Generator Loss
```
L_G = -E[D(fake)] + α · MSE(k2_fake, k2_real) + β · MSE(k3_fake, k3_real) - γ · H(fake)
```
- **Adversarial term:** -E[D(fake)]
- **K-mer matching:** Encourages the generator to reproduce biologically relevant 2-mer and 3-mer frequencies (α=2, β=5).
- **Entropy regularization:** -γ · H(fake) with γ=0.05 prevents the generator from collapsing to a single mode.

## Results

### RNA Generation (200 epochs)

| Metric | Real Data | Generated (Epoch 199) |
|--------|-----------|-----------------------|
| **GC Content** | 0.534 | 0.527 |
| **MFE (kcal/mol)** | -10.54 | -8.27 |
| **Avg. Length** | 51.06 | 45.50 |
| **Levenshtein Dist.** | — | 21.24 |

**Sample Sequences:**
```
UACUUCCCAUCGGAAUUGGGAAAC
ACAACACGGUGGCUCCCCUUGAAAGGUAAUAGCACC
GACAUGUCACCGCGGAUCGACGCCCUUCUCCGGCGAGGGCCGCC
AUGAGGACACUGGGGUUGACUUAAUUACACCUGGGCGCCAA
ACGCGAUGCCGGUAGCCUGGUUAUACAGAAGAUGAGUGUAGUCUGGGAAAAUACUUGAUUACUAAGUCUAUGUCCGGUCA
```

### DNA Generation (200 epochs)

| Metric | Real Data | Generated (Epoch 199) |
|--------|-----------|-----------------------|
| **GC Content** | 0.525 | 0.540 |
| **MFE (kcal/mol)** | -12.13 | -8.43 |
| **Avg. Length** | 52.83 | 42.58 |
| **Levenshtein Dist.** | — | 20.02 |

**Sample Sequences:**
```
TTCCTCCCCGTCGTAGGGCGGA
ACTATGGAGGCGAAGCAGTA
CGCCATTCAGCACTGCTGCACCTCATGACGTGCCATGGGCCGCG
TTGTGGTTAATG
ACGGGGTCACGACTGCCTTACCATTCTCATGTGGGGTCCGGGATGGAATGATACCTGATCAAAAGGTCAACGCCCACTCG
```

## Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- ViennaRNA Package (RNA module)
- pandas, numpy, Levenshtein

### Training
To train the RNA generator from scratch:
```bash
python rna_seq_gen.py
```

To train the DNA generator:
```bash
python dna_seq_gen.py
```

Training logs and metrics will be saved to `training_results/`.

## Key Design Decisions

1. **Why WGAN-GP?**  
   Standard GANs suffer from mode collapse and training instability. WGAN-GP provides a smooth loss landscape and meaningful metrics (Wasserstein distance).

2. **Why Gumbel-Softmax?**  
   Nucleotide sequences are discrete. Gumbel-Softmax allows differentiable sampling via the reparameterization trick, enabling end-to-end backpropagation.

3. **Why K-mer Loss?**  
   Biological function is often encoded in short motifs (2-3 bp). Matching k-mer distributions ensures the generator learns local sequence patterns, not just global GC content.

4. **Why Entropy Regularization?**  
   Prevents the generator from producing identical sequences (mode collapse) by maximizing the entropy of the output distribution.

## Future Work

- **Reinforcement Learning:** Use docking scores and MD stability metrics as rewards to fine-tune the generator.
- **Conditional Generation:** Condition the generator on a specific target protein to produce task-specific aptamers.
- **Progressive Growing:** Start with shorter sequences (20 bp) and gradually increase to 80 bp to improve long-range dependencies.


## Contact

For questions or collaboration: ebrilenkov@mail.ru