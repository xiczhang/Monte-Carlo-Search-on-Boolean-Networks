#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare ONLY the Uniform worst-case variance bound:
    V_wc = 1/(4M) + 1/(4 M n)
against the empirical variance of the ensemble estimator:
    p_hat_ens = (1/M) * sum_i p_hat_i
where each p_hat_i is estimated with n trajectories.

This runs the full BN simulator each repetition. Use small/medium grids if runtime matters.
"""

import os
import glob
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local modules expected in the environment
import bn_async_sim as bnas
import ensemble_utils as eu


# -------------------------
# Simulator config
# -------------------------

def make_sim_args(sims_per_model: int):
    """Common simulator kwargs (adjust phenotype if needed)."""
    return dict(
        output_nodes=["Apoptosis", "CellCycleArrest", "Invasion", "EMT"],
        sims_per_model=sims_per_model,
        max_steps=500,
        initial_activation={"miR200": 1, "miR203": 1, "miR34": 1, "DNAdamage": 0.5, "ECMicroenv": 0.5},
        sample_interval=1,
        threshold=0.0,
        target_pheno=["CellCycleArrest", "Invasion", "EMT"],  # phenotype tuple key
    )

def get_target_prob_from_hist(hist, target_tuple):
    """Extract final timepoint probability from ensemble history dict."""
    return float(hist[-1][1].get(target_tuple, 0.0))


# -------------------------
# Ensemble utilities
# -------------------------

def prepare_ensemble(zip_path: str, extract_dir: str):
    """Unzip (if needed) and load all .bnet models."""
    os.makedirs(extract_dir, exist_ok=True)
    if not os.listdir(extract_dir):
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Missing bundle: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    model_files = glob.glob(os.path.join(extract_dir, "*.bnet"))
    if not model_files:
        raise RuntimeError(f"No .bnet files found in {extract_dir}")

    networks = [bnas.load_network_model(f) for f in model_files]
    return networks


# -------------------------
# Core run
# -------------------------

def run_once_p_hat_ens(networks, M: int, n: int, rng: np.random.Generator) -> float:
    """
    One repetition: sample M networks WITH replacement, run n trajectories per model,
    and return the ensemble estimator \hat p_ens at final time.
    """
    sim_args = make_sim_args(sims_per_model=n)
    idx = rng.choice(len(networks), size=M, replace=True)
    chosen = [networks[k] for k in idx]

    # build_ensemble_history over the chosen models runs n per-model sims and aggregates
    hist = eu.build_ensemble_history(
        chosen,
        sim_args['output_nodes'],
        sim_args['sims_per_model'],
        sim_args['max_steps'],
        sim_args['initial_activation'],
        sim_args['sample_interval'],
        threshold=sim_args['threshold'],
        verbose=False
    )
    return get_target_prob_from_hist(hist, tuple(sim_args['target_pheno']))


def empirical_variance(run_fn, R: int) -> (float, np.ndarray):
    vals = np.array([run_fn() for _ in range(R)], dtype=float)
    return float(np.var(vals, ddof=1)), vals


# -------------------------
# Runner
# -------------------------

def run_experiment(
    zip_path: str,
    extract_dir: str,
    Ms: list,
    ns: list,
    R: int,
    seed: int,
    out_csv: str,
    out_json: str
):
    rng = np.random.default_rng(seed)
    networks = prepare_ensemble(zip_path, extract_dir)

    results = []
    pbar = tqdm(total=len(Ms) * len(ns), desc="Grid")

    for M in Ms:
        for n in ns:
            run_fn = lambda: run_once_p_hat_ens(networks, M, n, rng)
            V_emp, samples = empirical_variance(run_fn, R=R)

            # Uniform worst-case bound
            V_wc = 0.25 / M + 0.25 / (M * n)

            results.append({
                "M": M,
                "n": n,
                "R": R,
                "V_emp": V_emp,
                "V_wc": V_wc,
                "tight_wc": V_wc / V_emp,
                "cov_wc": int(V_wc >= V_emp),
                "sample_mean": float(np.mean(samples)),
                "sample_std": float(np.std(samples, ddof=1)),
            })
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({"grid": results}, f, indent=2)

    print(f"\n✅ Saved results to {out_csv}")
    print(f"📝 Saved JSON to {out_json}")


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Empirical variance vs. Uniform worst-case bound")
    p.add_argument("--zip_path", type=str, default="../data/bundle-exactpkn32-nocyclic-globalfps.zip")
    p.add_argument("--extract_dir", type=str, default="../data/WT_ensemble")

    p.add_argument("--Ms", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500, 1000])
    p.add_argument("--ns", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200, 500])
    p.add_argument("--R", type=int, default=200, help="repetitions per (M,n) cell")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="vb_wc_only.csv")
    p.add_argument("--out_json", type=str, default="vb_wc_only.json")
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(
        zip_path=args.zip_path,
        extract_dir=args.extract_dir,
        Ms=args.Ms,
        ns=args.ns,
        R=args.R,
        seed=args.seed,
        out_csv=args.out_csv,
        out_json=args.out_json
    )


if __name__ == "__main__":
    main()
