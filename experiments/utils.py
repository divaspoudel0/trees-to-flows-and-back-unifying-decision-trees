"""
Shared utilities for reproducing 'Trees to Flows and Back' (arXiv 2605.00414).
All experiments use only numpy, scipy, scikit-learn, and torch (CPU).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import (
    make_moons, make_blobs, make_classification,
    load_wine, load_breast_cancer, load_iris,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import wasserstein_distance

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_results(name: str, data: dict):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Results saved to {path}")
    return path


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)
