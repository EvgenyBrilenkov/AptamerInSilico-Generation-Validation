# 🧬 End-to-End AI-Driven Aptamer Discovery Pipeline

## Overview

This repository implements a complete computational pipeline for the **generation** and **in silico validation** of high-affinity aptamers. The project combines deep learning (WGAN-GP) for sequence generation with structural bioinformatics (folding, docking, molecular dynamics) for validation.

**Goal:** Replace costly, time-consuming experimental screening (SELEX) with an AI-driven workflow that generates novel aptamer candidates and rigorously validates their binding potential.

## 🏗️ Pipeline Architecture

The pipeline consists of two major components:

### 1. 🧪 Generation (WGAN-GP)
Generates novel DNA/RNA aptamer sequences using a modified Wasserstein GAN with Gradient Penalty.

**Key Features:**
- Gumbel-Softmax for discrete sequence sampling
- K-mer matching loss for biological motif preservation
- Entropy regularization to ensure diversity
- Upsample + Conv1d architecture to avoid artifacts

**Output:** A library of 1000+ chemically valid, diverse aptamer sequences.

📂 **Repository:** [aptamer-generation](./Aptamer-Generation/)

---

### 2. 🔬 Validation (In Silico Pipeline)
A 6-block automated workflow to assess the binding affinity and stability of generated aptamers.

**Blocks:**
1. **ML Screening & Consensus:** Predict binding probability using Apipred, AptaTrans, AptaNet.
2. **Protein Folding:** Obtain 3D structures via ESMFold or AlphaFold DB.
3. **Secondary Structure Prediction:** Predict aptamer 2D structure (dot-bracket) using RNAfold.
4. **3D Aptamer Modeling:** Generate tertiary structures using NSP assembly + refinement.
5. **Molecular Docking:** Predict protein-aptamer complexes with HDOCK.
6. **Coarse-Grained MD:** Validate complex stability using Martini 3 force field in GROMACS.

**Output:** Ranked list of aptamer candidates with docking scores and MD stability metrics.

📂 **Repository:** [aptamer-validation](./Aptamer-Validation/)

---

## 📂 Repository Structure

```
aptamer-discovery-pipeline/
├── aptamer-generation/          # Generative model (WGAN-GP)
│   ├── data/
│   ├── weights/
│   ├── training_results/
│   ├── seq_gen.py
│   ├── seq_gen_dna.py
│   ├── metrics.py
│   └── README.md
│
├── aptamer-validation/          # In silico validation pipeline
│   ├── block1_ml_screening/
│   ├── block2_protein_folding/
│   ├── block3_secondary_structure/
│   ├── block4_3d_modeling/
│   ├── block5_docking/
│   ├── block6_md_simulation/
│   └── README.md
│
├── results/                     # Final ranked candidates
│   ├── generated_sequences.csv
│   ├── docking_scores.csv
│   └── md_stability.csv
│
└── README.md                    # This file
```

### Analyze Results
Top candidates (based on docking score and MD stability) will be saved to `results/`.

## 📊 Results Summary

### Generation Performance

| Metric | RNA (Real) | RNA (Generated) | DNA (Real) | DNA (Generated) |
|--------|------------|-----------------|------------|-----------------|
| **GC Content** | 0.534 | 0.537 | 0.525 | 0.536 |
| **MFE (kcal/mol)** | -10.54 | -9.17 | -12.13 | -8.26 |
| **Avg. Length** | 51.06 | 47.09 | 52.83 | 42.74 |
| **Levenshtein Dist.** | — | 21.96 | — | 20.15 |

**Interpretation:**
- **GC Content:** Nearly identical to real data, ensuring chemical validity.
- **MFE:** Slightly higher (less negative) than real data, indicating good but not perfect thermodynamic stability. This will be improved via RL.
- **Length:** ~42-47 bp, which corresponds to the functional binding core of aptamers.
- **Levenshtein Distance:** ~20-22, confirming high sequence diversity (not memorization).

## 🧠 Technical Highlights

### Generation
- **WGAN-GP:** Stable training with meaningful loss metrics.
- **Gumbel-Softmax:** Differentiable discrete sampling with annealed temperature (`τ: 1.0 → 0.6`).
- **K-mer Loss:** Enforces biological motif similarity (2-mer and 3-mer frequencies).
- **Entropy Regularization:** Prevents mode collapse.

### Validation
- **Consensus ML:** Combines 3 independent models (Apipred, AptaTrans, AptaNet) for robust screening.
- **ESMFold:** State-of-the-art protein structure prediction.
- **NSP Assembly:** Generates multiple 3D conformations per aptamer to account for flexibility.
- **HDOCK:** Fast, accurate protein-RNA docking.
- **Martini 3 MD:** Coarse-grained simulation for long-timescale stability assessment.

## 🔮 Future Work

1. **Reinforcement Learning Loop:**
   - Use MD stability metrics (RMSD, RMSF, binding energy) as rewards.
   - Fine-tune the generator to produce sequences with higher predicted affinity.

2. **All-Atom MD:**
   - Transition from coarse-grained (Martini 3) to all-atom (AMBER/CHARMM) for top candidates.
   - Capture specific hydrogen bonds and electrostatic interactions.

3. **Experimental Validation:**
   - Synthesize top 5 candidates.
   - Test binding affinity in vitro (EMSA, SPR).

4. **Conditional Generation:**
   - Condition the generator on a specific target protein.
   - Produce task-specific aptamers for therapeutic applications.

## 📧 Contact

For questions or collaboration: ebrilenkov@mail.ru