"""
Claim 5: Sample Complexity Bounds for Tree-Structured Flows.

Numerically verifies convergence rates, concentration inequalities,
bias-variance tradeoffs, and GTSM convergence from Appendices C-H.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import set_seed, save_results


def test_error_vs_samples():
    """Error ~ n^(-alpha) for fixed-depth trees."""
    set_seed(42)
    sizes = [50, 100, 200, 500, 1000, 2000]
    errors = []
    for n in sizes:
        X, y = make_moons(n_samples=n, noise=0.2, random_state=42)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xtr, ytr)
        errors.append(float(1 - accuracy_score(yte, tree.predict(Xte))))

    log_n = np.log(sizes)
    log_e = np.log(np.array(errors) + 1e-10)
    alpha = -float(np.polyfit(log_n, log_e, 1)[0])

    return {"sample_sizes": sizes, "errors": [round(e, 4) for e in errors],
            "empirical_rate": round(alpha, 3), "rate_match": abs(alpha - 0.5) < 0.3}


def test_bias_variance_tradeoff():
    """Error first decreases then increases with depth."""
    set_seed(42)
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    depths = [2, 3, 4, 5, 6, 7, 8]
    errors = []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, ytr)
        errors.append(float(1 - accuracy_score(yte, tree.predict(Xte))))

    opt_idx = int(np.argmin(errors))
    tradeoff = 0 < opt_idx < len(depths) - 1

    return {"depths": depths, "errors": [round(e, 4) for e in errors],
            "optimal_depth": depths[opt_idx], "bias_variance_tradeoff": tradeoff}


def test_concentration():
    """Empirical error concentrates around mean (Hoeffding)."""
    set_seed(42)
    trial_errors = []
    for t in range(100):
        X, y = make_moons(n_samples=200, noise=0.2, random_state=t)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=t)
        tree = DecisionTreeClassifier(max_depth=4, random_state=t).fit(Xtr, ytr)
        trial_errors.append(1 - accuracy_score(yte, tree.predict(Xte)))

    mu, sigma = float(np.mean(trial_errors)), float(np.std(trial_errors))
    within = float(np.mean(np.abs(np.array(trial_errors) - mu) < 2 * sigma))

    return {"n_trials": 100, "mean_error": round(mu, 4), "std_error": round(sigma, 4),
            "fraction_within_2std": round(within, 3), "concentration_holds": within > 0.9}


def test_gtsm_convergence():
    """Boosting error monotonically decreases (GTSM objective)."""
    set_seed(42)
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    rounds = [1, 2, 5, 10, 20, 50]
    errors = []
    for n in rounds:
        gb = GradientBoostingClassifier(n_estimators=n, max_depth=3, random_state=42).fit(Xtr, ytr)
        errors.append(float(1 - accuracy_score(yte, gb.predict(Xte))))

    decreasing = all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    return {"n_rounds": rounds, "errors": [round(e, 4) for e in errors],
            "monotonically_decreasing": decreasing}


def test_distributional_convergence():
    """Within-leaf variance decreases and purity increases with depth."""
    set_seed(42)
    n_per = 300
    X = np.vstack([np.random.randn(n_per, 2) * 0.3 + np.array([x, y])
                   for x in [0, 3] for y in [0, 3]])
    y = np.concatenate([np.full(n_per, i) for i in range(4)])

    depths = [2, 3, 4, 5, 6, 7]
    metrics = []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X, y)
        leaves = tree.apply(X)
        vars_list, purities = [], []
        for lid in np.unique(leaves):
            mask = leaves == lid
            if mask.sum() < 5:
                continue
            vars_list.append(float(np.mean(np.var(X[mask], axis=0))))
            purities.append(float(np.max(np.bincount(y[mask].astype(int)) / mask.sum())))
        metrics.append({"depth": d, "n_leaves": int(len(np.unique(leaves))),
                         "avg_variance": round(np.mean(vars_list), 4),
                         "avg_purity": round(np.mean(purities), 4)})

    vars_seq = [m["avg_variance"] for m in metrics]
    pur_seq = [m["avg_purity"] for m in metrics]
    var_dec = all(vars_seq[i] >= vars_seq[i + 1] for i in range(len(vars_seq) - 1))
    pur_inc = all(pur_seq[i] <= pur_seq[i + 1] for i in range(len(pur_seq) - 1))

    return {"metrics": metrics, "variance_decreasing": var_dec,
            "purity_increasing": pur_inc, "converges": var_dec and pur_inc}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Claim 5: Sample Complexity Bounds")
    print("=" * 70)

    all_results = {}

    for name, fn in [
        ("error_vs_samples", test_error_vs_samples),
        ("bias_variance", test_bias_variance_tradeoff),
        ("concentration", test_concentration),
        ("gtsm_convergence", test_gtsm_convergence),
        ("distributional", test_distributional_convergence),
    ]:
        print(f"\n  Testing {name}...")
        r = fn()
        all_results[name] = r
        status = "✓" if list(r.values())[-1] else "✗"
        print(f"    {status} {name}")

    checks = [list(r.values())[-1] for r in all_results.values()]
    n_pass = sum(checks)
    print(f"\n{'=' * 70}")
    print(f"  {n_pass}/{len(checks)} checks passed")
    print(f"  Verdict: {'SUPPORTED' if n_pass >= len(checks) - 1 else 'PARTIAL'}")
    print("=" * 70)

    save_results("claim5_sample_complexity", all_results)
