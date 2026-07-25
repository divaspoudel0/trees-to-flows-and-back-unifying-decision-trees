"""
Run all experiments for reproducing 'Trees to Flows and Back' (arXiv 2605.00414).

Usage:
    python run_all.py              # Run all experiments
    python run_all.py --claim 1    # Run only Claim 1
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path

EXPERIMENTS = [
    (1, "claim1_theorems", "Tree-Flow Theorems 2.1-2.10"),
    (2, "claim2_dsm_tree", "DSmTree Distillation"),
    (3, "claim3_implicit_tree", "Implicit Tree Emergence"),
    (4, "claim4_treeflow", "TreeFlow Generation"),
    (5, "claim5_sample_complexity", "Sample Complexity Bounds"),
]


def run_experiment(module_path: Path, claim_num: int, description: str):
    print(f"\n{'#' * 70}")
    print(f"# Claim {claim_num}: {description}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(
        [sys.executable, str(module_path)],
        capture_output=True, text=True, timeout=600
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    if result.returncode != 0:
        print(f"  [ERROR] Claim {claim_num} failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", type=int, help="Run only this claim number (1-5)")
    args = parser.parse_args()

    base = Path(__file__).parent
    total_start = time.time()

    for num, folder, desc in EXPERIMENTS:
        if args.claim and args.claim != num:
            continue
        module_path = base / folder / "run.py"
        if not module_path.exists():
            print(f"  [SKIP] {module_path} not found")
            continue
        run_experiment(module_path, num, desc)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {elapsed:.1f}s")
    print(f"Results saved to: {base.parent / 'results'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
