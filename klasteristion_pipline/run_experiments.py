"""
Experiment runner — executes all experiments sequentially.

Usage
-----
    # Run all experiments on both pools with auto-detected CSV:
    python run_experiments.py

    # Specific CSV and pool:
    python run_experiments.py  path/to/reviews.csv  --pool negative

    # Run only specific experiment:
    python run_experiments.py  --exp 1
    python run_experiments.py  --exp 2 --model bge-m3
    python run_experiments.py  --exp 3 --model bge-m3 --nn 15 --mcs 10
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_UMAP_PARAMS, DEFAULT_HDBSCAN_PARAMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("experiments")


def main():
    parser = argparse.ArgumentParser(description="Run clustering experiments")
    parser.add_argument("csv", nargs="?", default=None, help="Path to CSV")
    parser.add_argument("--pool", default="negative", choices=["negative", "positive"])
    parser.add_argument("--exp", type=int, default=None, help="Run specific experiment (1/2/3)")
    parser.add_argument("--model", default="bge-m3", help="Embedding model for exp 2/3")
    parser.add_argument("--nn", type=int, default=None, help="UMAP n_neighbors for exp 3")
    parser.add_argument("--nc", type=int, default=None, help="UMAP n_components for exp 3")
    parser.add_argument("--mcs", type=int, default=None, help="HDBSCAN min_cluster_size for exp 3")
    args = parser.parse_args()

    t_start = time.perf_counter()
    experiments_to_run = [args.exp] if args.exp else [1, 2, 3]

    # ── Experiment 1 ─────────────────────────────────────────
    if 1 in experiments_to_run:
        print("\n" + "█" * 70)
        print("  EXPERIMENT 1 — Embedding Space Quality")
        print("█" * 70)
        from experiments.exp1_embeddings import run as run_exp1
        exp1_df = run_exp1(args.csv, args.pool)

        if len(exp1_df) and "silhouette_score" in exp1_df.columns:
            best_model = exp1_df.loc[exp1_df["silhouette_score"].idxmax(), "model"]
            log.info("Exp 1 winner: %s", best_model)
            if args.model == "bge-m3":
                args.model = best_model

    # ── Experiment 2 ─────────────────────────────────────────
    if 2 in experiments_to_run:
        print("\n" + "█" * 70)
        print(f"  EXPERIMENT 2 — UMAP & HDBSCAN Grid Search (model={args.model})")
        print("█" * 70)
        from experiments.exp2_umap_hdbscan import run as run_exp2
        exp2_df = run_exp2(args.csv, args.pool, args.model)

        if len(exp2_df):
            score_col = "composite_score" if "composite_score" in exp2_df.columns else "silhouette_score"
            viable = exp2_df[exp2_df["noise_pct"] <= 35] if "noise_pct" in exp2_df.columns else exp2_df
            if len(viable) == 0:
                viable = exp2_df
            if score_col in viable.columns and viable[score_col].notna().any():
                best = viable.loc[viable[score_col].idxmax()]
                args.nn = int(best["n_neighbors"])
                args.nc = int(best.get("n_components", 5))
                args.mcs = int(best["min_cluster_size"])
                log.info("Exp 2 winner: nn=%d, nc=%d, mcs=%d", args.nn, args.nc, args.mcs)

    # ── Experiment 3 ─────────────────────────────────────────
    if 3 in experiments_to_run:
        umap_p = dict(DEFAULT_UMAP_PARAMS)
        hdbscan_p = dict(DEFAULT_HDBSCAN_PARAMS)
        if args.nn:
            umap_p["n_neighbors"] = args.nn
        if args.nc:
            umap_p["n_components"] = args.nc
        if args.mcs:
            hdbscan_p["min_cluster_size"] = args.mcs

        print("\n" + "█" * 70)
        print(f"  EXPERIMENT 3 — Topic Coherence (model={args.model}, nn={umap_p['n_neighbors']}, nc={umap_p['n_components']}, mcs={hdbscan_p['min_cluster_size']})")
        print("█" * 70)
        from experiments.exp3_topic_coherence import run as run_exp3
        run_exp3(args.csv, args.pool, args.model, umap_p, hdbscan_p)

    # ── Done ─────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    print("\n" + "█" * 70)
    print(f"  ALL EXPERIMENTS COMPLETE — {total:.1f} s total")
    print("█" * 70)


if __name__ == "__main__":
    main()
