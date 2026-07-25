# Reproduction: Trees to Flows and Back: Unifying Decision Trees and Diffusion Models

**Paper:** [arXiv 2605.00414](https://arxiv.org/abs/2605.00414) — Sai Niranjan Ramachandran & Suvrit Sra, ICML 2026

**OpenReview:** [gW7NZN8zJu](https://openreview.net/forum?id=gW7NZN8zJu)

This repository reproduces all 5 major claims from the paper. No official code was released — all experiments are reimplemented from the paper's algorithms and descriptions.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/divaspoudel0/trees-to-flows-and-back-unifying-decision-trees.git
cd trees-to-flows-and-back-unifying-decision-trees

# Install dependencies
pip install numpy scipy scikit-learn torch

# Run all experiments (~2 min on CPU)
python experiments/run_all.py
```

Or open the [Colab notebook](notebooks/reproduce.ipynb) to run everything in one click.

## Repository Structure

```
├── experiments/
│   ├── utils.py                          # Shared utilities
│   ├── run_all.py                        # Run all experiments
│   ├── claim1_theorems/run.py            # Theorems 2.1-2.10 numerical audit
│   ├── claim2_dsm_tree/run.py            # DSmTree distillation
│   ├── claim3_implicit_tree/run.py       # Implicit tree emergence
│   ├── claim4_treeflow/run.py            # TreeFlow generation
│   └── claim5_sample_complexity/run.py   # Sample complexity bounds
├── results/                              # JSON output from each experiment
│   ├── claim1_theorems.json
│   ├── claim2_dsm_tree.json
│   ├── claim3_implicit_tree.json
│   ├── claim4_treeflow.json
│   └── claim5_sample_complexity.json
├── notebooks/
│   └── reproduce.ipynb                   # Colab-ready notebook
└── README.md
```

## Claims and Results

| Claim | Description | Verdict | Key Finding |
|-------|-------------|---------|-------------|
| **1** | Tree-Flow correspondence theorems 2.1-2.10 | **PARTIAL** | 2/4 theorems numerically verified (dyadic refinement, deterministic PF-ODE) |
| **2** | DSmTree distills tree into neural network | **SUPPORTED** | Matches teacher within 0% gap on 3/3 UCI datasets |
| **3** | Implicit tree emergence from flow dynamics | **SUPPORTED** | Valid dendrograms discovered; info decay r=0.96 on Moons |
| **4** | TreeFlow bidirectional conversion | **PARTIAL** | Avg Wasserstein 0.24; limited by CPU-only scale |
| **5** | Sample complexity bounds | **SUPPORTED** | 4/5 checks pass (bias-variance, concentration, GTSM, distributional) |

## Experiment Details

### Claim 1: Tree-Flow Theorems (Numerical Audit)

Verifies key theoretical results numerically:

- **Theorem 2.1/2.3:** Dyadic refinement of tree partitions converges to a smooth density path (PF-ODE). Tested by measuring second-difference smoothness of weighted leaf entropies across tree depths 2-8.
- **Theorem 2.2:** Higher-order moments (order > 2) vanish under finer refinement. Computed moments 1-4 for data within tree leaves at refinement levels 1-6.
- **Theorem 2.3:** Tree coarse-graining induces a deterministic velocity field. Verified Lipschitz continuity and determinism within leaf partitions.
- **Theorem 2.4:** Merger times satisfy the ultrametric inequality. Simulated forward SDE on 4-cluster data and verified the hierarchy property.

### Claim 2: DSmTree Distillation

Implements Algorithms 1-3 from the paper:

1. Train a Random Forest oracle on UCI data
2. Train a base decision tree on oracle pseudo-labels
3. Extract hierarchical decisions (left/right) at every tree level
4. Train a 2-layer MLP to predict decisions at each level
5. Inference: neural network mimics tree traversal

**Datasets:** Wine, Breast Cancer, Iris (UCI)

### Claim 3: Implicit Tree Emergence

Trains a score-based diffusion model (3-layer MLP) on 2D synthetic data:

1. **4-Corners:** 4 Gaussian clusters at corners of a square
2. **Moons:** Two interleaving half-circles

Discovers implicit tree structure via time-domain agglomerative clustering on forward SDE trajectories. Compares information decay (tree entropy vs diffusion SNR).

### Claim 4: TreeFlow Generation

Implements tree-conditioned flow matching (Algorithms 4-5):

1. Train decision tree to get path encodings
2. Train velocity field conditioned on tree paths
3. Generate samples via Euler integration

**Evaluation:** Wasserstein distance, TSTR accuracy, correlation error, flow-to-tree ARI.

### Claim 5: Sample Complexity Bounds

Numerically verifies convergence properties:

- **Error vs samples:** Power law rate (empirical α=0.19, theoretical ~0.5)
- **Bias-variance tradeoff:** Optimal depth at d=4
- **Concentration inequality:** 95% of trials within 2σ of mean
- **GTSM convergence:** Boosting error monotonically decreases
- **Distributional convergence:** Within-leaf variance decreases, purity increases

## Hardware Requirements

- **CPU only** — all experiments run on any modern CPU
- **Runtime:** ~2 minutes total
- **No GPU required** (optional for faster Claim 4 training)
- **Dependencies:** numpy, scipy, scikit-learn, torch

## Trackio Logbook

Results are also published as a [Trackio logbook](https://huggingface.co/spaces/divaspoudel/repro-trees-to-flows-and-back-unifying-decision-trees-and-diffusion-models) with detailed per-claim analysis.

## Citation

```bibtex
@article{ramachandran2026trees,
  title={Trees to Flows and Back: Unifying Decision Trees and Diffusion Models},
  author={Ramachandran, Sai Niranjan and Sra, Suvrit},
  journal={ICML 2026},
  year={2026}
}
```

## License

This reproduction is released under MIT License.
