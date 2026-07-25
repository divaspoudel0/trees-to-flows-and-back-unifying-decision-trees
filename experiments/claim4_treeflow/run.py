"""
Claim 4: TreeFlow — Tree-Conditioned Flow Matching for Tabular Generation.

Implements TreeFlow (Algorithm 4-5) and tests bidirectional flow↔tree conversion.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wasserstein_distance
from utils import set_seed, save_results


# ── Path Encoder ──────────────────────────────────────────────────────────────

class PathEncoder(nn.Module):
    def __init__(self, max_depth, dim=16):
        super().__init__()
        self.emb = nn.Embedding(max_depth + 1, dim)

    def forward(self, paths):
        return self.emb(paths).mean(dim=1)


# ── Velocity Network ─────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    def __init__(self, input_dim, max_depth, enc_dim=16, hidden=128):
        super().__init__()
        self.path_enc = PathEncoder(max_depth, enc_dim)
        self.time_enc = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.net = nn.Sequential(
            nn.Linear(input_dim + 32 + enc_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x, t, paths):
        return self.net(torch.cat([x, self.time_enc(t), self.path_enc(paths)], dim=-1))


# ── Tree Path Extraction ─────────────────────────────────────────────────────

def extract_paths(tree, X, max_depth):
    paths = np.zeros((len(X), max_depth), dtype=int)
    ts = tree.tree_
    for i in range(len(X)):
        node = 0
        for d in range(max_depth):
            if ts.children_left[node] == ts.children_right[node]:
                break
            if X[i, ts.feature[node]] <= ts.threshold[node]:
                paths[i, d] = 0
                node = ts.children_left[node]
            else:
                paths[i, d] = 1
                node = ts.children_right[node]
    return paths


# ── Train TreeFlow ────────────────────────────────────────────────────────────

def train_treeflow(X_train, max_depth=4, epochs=400, lr=1e-3, batch=128):
    input_dim = X_train.shape[1]
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X_train, np.zeros(len(X_train)))
    paths = extract_paths(tree, X_train, max_depth)
    n_leaves = len(np.unique(tree.apply(X_train)))

    model = VelocityNet(input_dim, max_depth, hidden=128)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.FloatTensor(X_train)
    Pt = torch.LongTensor(paths)
    ds = TensorDataset(Xt, Pt)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for xb, pb in loader:
            n = xb.shape[0]
            t = torch.rand(n, 1)
            noise = torch.randn_like(xb)
            x_t = t * xb + (1 - t) * noise
            v_target = xb - noise
            v_pred = model(x_t, t, pb)
            loss = torch.mean((v_pred - v_target) ** 2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        losses.append(ep_loss / len(loader))
        if (ep + 1) % 100 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={losses[-1]:.4f}")

    return model, tree, losses, n_leaves


# ── Generate Samples ──────────────────────────────────────────────────────────

def generate(model, n_samples, max_depth, input_dim, n_steps=50):
    model.eval()
    x = torch.randn(n_samples, input_dim)
    paths = torch.randint(0, 2, (n_samples, max_depth))
    dt = 1.0 / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((n_samples, 1), i * dt)
            x = x + model(x, t, paths) * dt
    return x.numpy()


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(name, data, max_depth=4):
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)

    print(f"\n  [{name}] Training TreeFlow...")
    t0 = time.time()
    model, tree, losses, n_leaves = train_treeflow(Xtr, max_depth=max_depth)
    train_time = time.time() - t0

    print("    Generating samples...")
    t0 = time.time()
    X_gen = generate(model, len(Xte), max_depth, Xtr.shape[1])
    gen_time = time.time() - t0

    # Wasserstein distance
    wd = float(np.mean([wasserstein_distance(Xte[:, d], X_gen[:, d]) for d in range(Xtr.shape[1])]))

    # TSTR accuracy
    clf = RandomForestClassifier(50, random_state=42).fit(X_gen, ytr[:len(X_gen)])
    tstr = float(accuracy_score(yte, clf.predict(Xte)))

    # Correlation error
    if Xtr.shape[1] > 1:
        corr_err = float(np.mean(np.abs(np.corrcoef(Xte.T) - np.corrcoef(X_gen.T))))
    else:
        corr_err = 0.0

    # Flow → Tree (agglomerative clustering on generated data)
    n_true = len(np.unique(y))
    clustering = AgglomerativeClustering(n_clusters=n_true).fit(X_gen[:min(500, len(X_gen))])
    ari = float(adjusted_rand_score(yte[:min(500, len(X_gen))], clustering.labels_))

    return {
        "wasserstein": round(wd, 4),
        "tstr_accuracy": round(tstr, 4),
        "correlation_error": round(corr_err, 4),
        "flow_to_tree_ari": round(ari, 4),
        "train_time_s": round(train_time, 2),
        "gen_time_s": round(gen_time, 3),
        "n_leaves": n_leaves,
        "final_loss": round(float(losses[-1]), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Claim 4: TreeFlow — Tree-Conditioned Flow Matching")
    print("=" * 70)
    set_seed(42)

    datasets = {"Wine": load_wine(), "Breast Cancer": load_breast_cancer(), "Iris": load_iris()}
    all_results = {}
    for name, data in datasets.items():
        all_results[name] = evaluate(name, data)

    print(f"\n{'=' * 70}")
    print("Results:")
    for name, r in all_results.items():
        print(f"  {name}: WD={r['wasserstein']:.3f}  TSTR={r['tstr_accuracy']:.1%}  ARI={r['flow_to_tree_ari']:.3f}  "
              f"train={r['train_time_s']:.1f}s  gen={r['gen_time_s']:.3f}s")

    avg_wd = np.mean([r["wasserstein"] for r in all_results.values()])
    avg_tstr = np.mean([r["tstr_accuracy"] for r in all_results.values()])
    avg_ari = np.mean([r["flow_to_tree_ari"] for r in all_results.values()])
    print(f"\n  Avg: WD={avg_wd:.3f}  TSTR={avg_tstr:.1%}  ARI={avg_ari:.3f}")
    print("=" * 70)

    save_results("claim4_treeflow", all_results)
