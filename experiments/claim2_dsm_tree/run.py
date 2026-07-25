"""
Claim 2: DSmTree — Distilling Hierarchical Tree Logic into Neural Networks.

Implements Algorithms 1-3 from the paper and verifies DSmTree matches
teacher performance within 2% on UCI benchmarks.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import set_seed, save_results


# ── Neural Network ────────────────────────────────────────────────────────────

class DSMTreeNet(nn.Module):
    """Predicts the binary decision (left/right) at each tree depth level."""

    def __init__(self, input_dim, max_depth, hidden=128):
        super().__init__()
        self.max_depth = max_depth
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(max_depth)])

    def forward(self, x, level):
        feat = self.features(x)
        out = torch.zeros(x.size(0), 2, device=x.device)
        for d in range(self.max_depth):
            mask = level == d
            if mask.any():
                out[mask] = self.heads[d](feat[mask])
        return out


# ── Tree decision extraction ─────────────────────────────────────────────────

def extract_tree_decisions(tree, X, max_depth):
    """Extract binary decision at each level for every sample. -1 = leaf reached."""
    n = X.shape[0]
    decisions = np.full((n, max_depth), -1, dtype=int)
    ts = tree.tree_
    for i in range(n):
        node = 0
        for d in range(max_depth):
            if ts.children_left[node] == ts.children_right[node]:
                break
            feat, thresh = ts.feature[node], ts.threshold[node]
            decisions[i, d] = 0 if X[i, feat] <= thresh else 1
            node = ts.children_left[node] if X[i, feat] <= thresh else ts.children_right[node]
    return decisions


# ── Training ──────────────────────────────────────────────────────────────────

def train_dsm_tree(X_train, y_train, depth=5, epochs=400, lr=2e-3, batch=128):
    """Full DSmTree training pipeline (Algorithms 1-2)."""
    input_dim = X_train.shape[1]

    # 1. Oracle → pseudo-labels
    oracle = RandomForestClassifier(n_estimators=100, random_state=42)
    oracle.fit(X_train, y_train)
    pseudo = oracle.predict(X_train)

    # 2. Base tree on pseudo-labels
    base_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    base_tree.fit(X_train, pseudo)

    # 3. Extract hierarchical decisions
    decisions = extract_tree_decisions(base_tree, X_train, depth)

    # 4. Build training data: (x, level) → decision
    all_X, all_L, all_D = [], [], []
    Xt = torch.FloatTensor(X_train)
    for i in range(len(X_train)):
        for d in range(depth):
            if decisions[i, d] >= 0:
                all_X.append(Xt[i])
                all_L.append(d)
                all_D.append(decisions[i, d])

    ds = TensorDataset(torch.stack(all_X), torch.LongTensor(all_L), torch.LongTensor(all_D))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    model = DSMTreeNet(input_dim, depth, hidden=128)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for xb, lb, db in loader:
            opt.zero_grad()
            loss = crit(model(xb, lb), db)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        losses.append(ep_loss / len(loader))
        if (ep + 1) % 100 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={losses[-1]:.5f}")

    return model, base_tree, losses


# ── Inference (Algorithm 3) ──────────────────────────────────────────────────

def dsm_tree_inference(model, base_tree, X, depth):
    """Neural network mimics tree traversal."""
    model.eval()
    ts = base_tree.tree_
    with torch.no_grad():
        Xr = torch.FloatTensor(X)
        nodes = np.zeros(len(X), dtype=int)
        reached = np.zeros(len(X), dtype=bool)

        for d in range(depth):
            active = (~reached) & (ts.children_left[nodes] != ts.children_right[nodes])
            if not active.any():
                break
            logits = model(Xr[active], torch.full((active.sum(),), d, dtype=torch.long))
            preds = logits.argmax(dim=1).numpy()
            for idx, p in zip(np.where(active)[0], preds):
                nodes[idx] = ts.children_left[nodes[idx]] if p == 0 else ts.children_right[nodes[idx]]
                if ts.children_left[nodes[idx]] == ts.children_right[nodes[idx]]:
                    reached[idx] = True

        return ts.value[nodes][:, 0, :].argmax(axis=1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name, data, depth=5):
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)

    print(f"\n  [{name}] Training DSmTree (depth={depth})...")
    model, tree, losses = train_dsm_tree(Xtr, ytr, depth=depth)

    teacher_acc = accuracy_score(yte, tree.predict(Xte))
    dsm_acc = accuracy_score(yte, dsm_tree_inference(model, tree, Xte, depth))
    oracle_acc = accuracy_score(yte, RandomForestClassifier(100, random_state=42).fit(Xtr, ytr).predict(Xte))
    gap = abs(teacher_acc - dsm_acc) * 100

    return {
        "teacher_accuracy": round(float(teacher_acc), 4),
        "dsm_accuracy": round(float(dsm_acc), 4),
        "oracle_accuracy": round(float(oracle_acc), 4),
        "gap_percent": round(float(gap), 2),
        "matches_teacher_2pct": gap <= 2.0,
        "final_loss": round(float(losses[-1]), 6),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Claim 2: DSmTree — Distilling Trees into Neural Networks")
    print("=" * 70)
    set_seed(42)

    datasets = {"Wine": load_wine(), "Breast Cancer": load_breast_cancer(), "Iris": load_iris()}
    all_results = {}
    for name, data in datasets.items():
        all_results[name] = evaluate(name, data)

    print(f"\n{'=' * 70}")
    print("Results:")
    for name, r in all_results.items():
        status = "✓" if r["matches_teacher_2pct"] else "✗"
        print(f"  {status} {name}: teacher={r['teacher_accuracy']:.1%}  dsm={r['dsm_accuracy']:.1%}  gap={r['gap_percent']:.1f}%")

    n_match = sum(r["matches_teacher_2pct"] for r in all_results.values())
    print(f"\n  {n_match}/{len(all_results)} datasets match teacher within 2%")
    print("=" * 70)

    save_results("claim2_dsm_tree", all_results)
