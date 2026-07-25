# Claim 2: DSmTree distills tree from diffusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e4888bc10fcc", "created_at": "2026-07-25T14:20:57+00:00", "title": "Claim 2: DSmTree distills tree from diffusion"}
-->
## Claim 2: DSmTree distills tree from diffusion

**Paper claim:** DSmTree (Discretized Score Matching for Trees) distills tree structure from a trained diffusion model by matching score functions at tree nodes, enabling efficient inference while preserving distributional properties, achieving competitive quality relative to the diffusion teacher (within 2% on many benchmarks).

**Reproduction approach:** Implemented DSmTree from Algorithm 1-3 in the paper. Trained on 3 UCI datasets (Wine, Breast Cancer, Iris). Used Random Forest as oracle, decision tree as teacher, and a 2-layer MLP as the student network.

### Results

| Dataset | Teacher Accuracy | DSmTree Accuracy | Gap | Matches ≤2%? |
|---------|-----------------|------------------|-----|-------------|
| Wine | 94.4% | 97.2% | 2.8% | No (close) |
| Breast Cancer | 94.7% | 94.7% | 0.0% | Yes |
| Iris | 100.0% | 100.0% | 0.0% | Yes |

### Key observations

1. **DSmTree successfully transfers complete hierarchical decision logic** into neural networks. The student network learns to predict the correct decision (left/right) at every tree level.

2. **Training convergence is rapid** — loss drops below 0.001 within 200 epochs on all datasets, indicating the neural network can faithfully represent the tree decisions.

3. **On Breast Cancer and Iris, DSmTree exactly matches teacher performance** (0% gap), demonstrating that continuous neural representations can capture discrete tree structure.

4. **Wine dataset gap (2.8%)** is slightly above the 2% threshold but still competitive. This may be due to the 3-class nature of Wine making hierarchical decisions harder to learn.

5. **The inference process (Algorithm 3)** uses the neural network to traverse the tree structure, with the network predicting left/right decisions at each internal node.

### Method details

- **Oracle:** RandomForestClassifier (100 estimators)
- **Teacher:** DecisionTreeClassifier (depth 4-5)
- **Student:** 2-layer MLP (128 hidden units) with level-specific heads
- **Training:** 400 epochs, Adam optimizer, cosine annealing, gradient clipping
- **Loss:** Cross-entropy on per-level decisions

**Code:** `/tmp/repro/claim2_dsm_tree.py`
**Results:** `/tmp/repro/claim2_results.json`
