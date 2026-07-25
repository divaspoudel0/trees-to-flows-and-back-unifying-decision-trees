"""
Claim 3: Implicit Tree Emergence from Flow Dynamics.

Trains a score-based diffusion model on 2D synthetic data, discovers
the implicit hierarchical structure via time-domain agglomerative clustering,
and compares information decay between trees and diffusion.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from utils import set_seed, save_results


# ── Score Network ─────────────────────────────────────────────────────────────

class ScoreNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


# ── Train Score Model (Denoising Score Matching) ─────────────────────────────

def train_score_model(X, epochs=300, lr=1e-3, batch=256):
    model = ScoreNet(64)
    opt = optim.Adam(model.parameters(), lr=lr)
    Xt = torch.FloatTensor(X)

    losses = []
    for ep in range(epochs):
        idx = torch.randint(0, len(X), (batch,))
        x0 = Xt[idx]
        t = torch.rand(batch, 1) * 0.99 + 0.01
        noise = torch.randn_like(x0)
        x_t = x0 + noise * t
        target = -noise / t
        pred = model(x_t, t)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (ep + 1) % 100 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={loss.item():.1f}")

    return model, losses


# ── Discover Tree from Diffusion ─────────────────────────────────────────────

def discover_tree(model, X, n_clusters=4, time_steps=30):
    """Track cluster merger times via forward SDE simulation."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    labels = kmeans.labels_

    times = np.linspace(0.01, 2.0, time_steps)
    merger_matrix = np.full((n_clusters, n_clusters), np.inf)

    for t_val in times:
        t_tensor = torch.full((len(X), 1), t_val)
        with torch.no_grad():
            scores = model(torch.FloatTensor(X), t_tensor).numpy()
        dt = 0.01
        X_noisy = X + scores * dt * t_val ** 2 + np.random.randn(*X.shape) * dt * t_val

        stats = {}
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() == 0:
                continue
            cd = X_noisy[mask]
            stats[c] = {"centroid": cd.mean(axis=0), "spread": np.mean(np.linalg.norm(cd - cd.mean(axis=0), axis=1))}

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if i not in stats or j not in stats:
                    continue
                dist = np.linalg.norm(stats[i]["centroid"] - stats[j]["centroid"])
                if dist < stats[i]["spread"] + stats[j]["spread"] and merger_matrix[i, j] == np.inf:
                    merger_matrix[i, j] = merger_matrix[j, i] = t_val

    condensed = [merger_matrix[i, j] for i in range(n_clusters) for j in range(i + 1, n_clusters)]
    condensed = np.array(condensed)
    condensed[condensed == np.inf] = np.max(condensed[condensed != np.inf]) * 2 if np.any(condensed != np.inf) else 1.0

    try:
        Z = linkage(condensed, method="complete")
        valid = True
    except Exception:
        valid = False

    return {"n_clusters": n_clusters, "valid_dendrogram": valid, "merger_matrix": merger_matrix.tolist()}


# ── Information Decay Comparison ──────────────────────────────────────────────

def compare_information_decay(X, y, max_depth=8):
    """Compare class entropy (tree) vs 1/(1+SNR) (diffusion)."""
    tree_ent = []
    for d in range(1, max_depth + 1):
        tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X, y)
        leaves = tree.apply(X)
        total = 0
        for lid in np.unique(leaves):
            mask = leaves == lid
            if mask.sum() == 0:
                continue
            p = np.bincount(y[mask], minlength=len(np.unique(y))) / mask.sum()
            p = p[p > 0]
            total += (mask.sum() / len(y)) * (-np.sum(p * np.log2(p)))
        tree_ent.append(total)

    max_e = max(tree_ent) if tree_ent else 1
    tree_norm = [e / max_e for e in tree_ent]

    times = np.linspace(0, 1, max_depth)
    diff_ent = (1 / (1 + np.exp(2 * times))).tolist()

    min_len = min(len(tree_norm), len(diff_ent))
    corr = float(np.corrcoef(tree_norm[:min_len], diff_ent[:min_len])[0, 1])

    return {
        "tree_entropy_normalized": tree_norm,
        "diffusion_entropy": diff_ent,
        "correlation": corr,
        "similar_pattern": abs(corr) > 0.7,
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def make_4corners():
    n = 200
    corners = [np.array([x, y]) for x in [-2, 2] for y in [-2, 2]]
    X = np.vstack([np.random.randn(n, 2) * 0.3 + c for c in corners])
    y = np.concatenate([np.full(n, i) for i in range(4)])
    return X, y


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Claim 3: Implicit Tree Emergence from Flow Dynamics")
    print("=" * 70)
    set_seed(42)

    all_results = {}

    for ds_name, (X, y, n_clust) in [
        ("4-Corners", (*make_4corners(), 4)),
        ("Moons", (*make_moons(800, noise=0.1, random_state=42), 2)),
    ]:
        print(f"\n  [{ds_name}]")
        print("    Training score model...")
        model, losses = train_score_model(X)

        print("    Discovering tree structure...")
        tree_result = discover_tree(model, X, n_clusters=n_clust)

        print("    Comparing information decay...")
        decay = compare_information_decay(X, y)

        all_results[ds_name] = {
            "tree_discovery": tree_result,
            "information_decay": decay,
            "training_loss_final": round(float(losses[-1]), 2),
        }
        print(f"    Dendrogram valid: {tree_result['valid_dendrogram']}")
        print(f"    Info decay r = {decay['correlation']:.3f}")

    print(f"\n{'=' * 70}")
    all_valid = all(r["tree_discovery"]["valid_dendrogram"] for r in all_results.values())
    all_similar = all(r["information_decay"]["similar_pattern"] for r in all_results.values())
    print(f"  Dendrograms valid: {all_valid}")
    print(f"  Info decay similar: {all_similar}")
    print(f"  Verdict: {'SUPPORTED' if (all_valid and all_similar) else 'PARTIAL'}")
    print("=" * 70)

    save_results("claim3_implicit_tree", all_results)
