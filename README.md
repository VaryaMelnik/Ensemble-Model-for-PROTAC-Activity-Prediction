# Ensemble Model for PROTAC Activity Prediction

A heterogeneous stacking ensemble classifier for predicting the biological 
activity of PROTAC (Proteolysis Targeting Chimera) molecules. The model was 
developed as a scoring function for reinforcement-learning-driven linker 
design within a generative PROTAC discovery pipeline.

## Overview

The ensemble integrates four base classifiers, each operating on a distinct 
molecular representation, and aggregates their predictions through a 
logistic regression meta-classifier. This design exploits the complementarity 
between heterogeneous feature spaces — topological, pharmacophoric, semantic, 
and protein-context — to produce a more reliable scoring signal than any 
single representation can provide.

## Architecture


| Base classifier | Feature representation                                              |
|-----------------|---------------------------------------------------------------------|
| XGBoost         | Pharmacophore fingerprints + protein embeddings                     |
| Random Forest   | Morgan fingerprints (ECFP) + protein embeddings                     |
| SVM             | RDKit 2D physicochemical descriptors + protein embeddings           |
| KNN             | ChemBERTa embeddings + protein embeddings               |
| **Meta-model**  | Logistic regression over base classifier predictions                |

## Performance

Evaluated on a held-out test set never seen during training or hyperparameter
tuning:

| Metric       | Value |
|--------------|-------|
| MCC          | 0.75  |
| ROC-AUC      | 0.92  |
| AUPRC        | 0.83  |
| Accuracy     | 0.92  |
| Precision    | 0.87  |
| Recall       | 0.74  |
| F1           | 0.80  |
| Specificity  | 0.97  |

Validity was further confirmed via Y-randomization and epistemic uncertainty 
analysis. Combinatorial ablation across all 15 subsets of base 
classifiers confirmed that the full four-model ensemble yields the highest 
MCC, with a 13.8% improvement over the best individual classifier.

## Data

Training data was compiled from PROTAC-DB. Activity labels were assigned 
using a hierarchical three-level annotation scheme adapted from [CITATION]:

**Level I — Functional degradation** (preferred when available).  
A compound is labeled active if DC50 ≤ 1000 nM or Dmax ≥ 70%.

**Level II — Ternary complex stability** (used when Level I data are 
unavailable).  
A compound is labeled active if Kd ≤ 500 nM, t½ ≥ 30 s, 
or ΔG ≤ −8 kcal/mol.

**Level III — Binary target affinity** (used when Levels I and II are 
unavailable).  
A compound is labeled active if it shows experimentally confirmed affinity 
(EC50, Kd, or Ki ≤ 1000 nM) to **both** the target protein and the E3 ligase 
simultaneously.

Compounds that do not meet the criteria of any applicable level, or that 
lack sufficient experimental data, are labeled inactive.

This hierarchical scheme reflects the biological causality of PROTAC action: 
functional degradation is the most direct readout of activity, ternary 
complex stability is its proximal mechanistic prerequisite, and dual binary 
affinity is the minimal necessary (but not sufficient) condition. Lower 
levels are used only as fallback when higher-level data are unavailable, 
which maximizes data utilization while preserving label fidelity.

## Intended Use

This model is designed as a **prioritization tool**, not a quantitative 
predictor of degradation activity. It is intended for:

- Scoring candidates during generative linker design (e.g., as a reward 
  component in REINVENT4 / Link-INVENT)
- Triaging large virtual libraries of PROTAC candidates
- Ranking molecules for downstream structural validation (docking, MD)

It is **not** intended to replace experimental measurement of DC50, Dmax, 
or ternary complex cooperativity.

## Installation

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
```
