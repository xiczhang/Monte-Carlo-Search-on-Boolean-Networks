import os
import sys
import time
import glob
import json
import zipfile
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import math
# Local modules
import src.bn_async_sim as bnas
import src.ensemble_utils as eu
from src.nmcs_module import nmcs
from src.lnmcs_module import lnmcs
from src.nrpa_module import nrpa
from src.gnrpa_module import gnrpa


# =====================
# Simulation Utilities
# =====================

def adaptive_sims_per_model(ensemble_size: int) -> int:
    return max(1, int(math.ceil(2500 / ensemble_size))) # simulation noise = 0.01 * 0.01=1/10000

def make_sim_args(ensemble_size: int):
    return dict(
        output_nodes=["Apoptosis", "CellCycleArrest", "Invasion", "EMT"],
        sims_per_model=adaptive_sims_per_model(ensemble_size),
        max_steps=500,
        initial_activation={"miR200": 1, "miR203": 1, "miR34": 1, "DNAdamage": 0.5, "ECMicroenv": 0.5},
        sample_interval=1,
        threshold=0.0,
        target_pheno=["CellCycleArrest", "Invasion", "EMT"]
    )

class EvalCounter:
    def __init__(self, networks, ensemble_size):
        self.networks = networks
        self.sim_args = make_sim_args(ensemble_size)
        self.count = 0
        self._cache = {}

    def evaluate(self, muts):
        self.count += 1
        key = tuple(sorted(muts))
        if key in self._cache:
            return self._cache[key]
        mutated = [bnas.create_mutant_network(net, muts) for net in self.networks]
        hist = eu.build_ensemble_history(
            mutated,
            self.sim_args['output_nodes'],
            self.sim_args['sims_per_model'],
            self.sim_args['max_steps'],
            self.sim_args['initial_activation'],
            self.sim_args['sample_interval'],
            threshold=self.sim_args['threshold'],
            verbose=False
        )
        score = hist[-1][1].get(tuple(self.sim_args['target_pheno']), 0.0)
        self._cache[key] = score
        return score

    def clear_cache(self):
        self._cache.clear()
        self.count = 0

def prepare_ensemble(zip_path: str, extract_dir: str, ensemble_size_max: int):
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
    random.Random(2).shuffle(networks)

    def subset(n): return networks[:min(n, len(networks))]

    sizes = [50, 200, 500, 700, 1000]
    subsets = {s: subset(s) for s in sizes if s <= len(networks)}
    if 1000 not in subsets and ensemble_size_max > 1000:
        subsets[1000] = subset(ensemble_size_max)
    return networks, subsets

def build_all_moves(network, sim_args):
    raw = eu.generate_single_mutants(network, sim_args["output_nodes"])
    return [tuple(m[0]) if isinstance(m, list) else tuple(m) for m in raw]

def make_evaluators(subsets):
    ec_map = {k: EvalCounter(v, k) for k, v in subsets.items()}
    ec_fast = ec_map.get(50, ec_map[min(ec_map.keys())])
    return ec_map, ec_fast


# =====================
# Experiment Execution
# =====================

def run_single(algo, depth, timeout, all_moves, ec, ec_fast):
    if algo == "NMCS":
        return nmcs([], level=2, depth=depth, best_moves=all_moves, ec=ec, timeout_sec=timeout)
    if algo == "LNMCS":
        return lnmcs([], level=2, depth=depth, all_moves=all_moves, ec=ec, b=3, r=0.4, e=10, timeout_sec=timeout)
    if algo == "NRPA":
        return nrpa(level=2, policy={}, depth=depth, ec=ec, all_moves=all_moves, timeout_sec=timeout)
    if algo == "GNRPA":
        score, mut_set, *_ = gnrpa(level=2, policy={}, bias={}, tau=0.5, depth=depth, ec=ec, all_moves=all_moves, N=100, timeout_sec=timeout)
        return score, mut_set
    raise ValueError(f"Unknown algorithm: {algo}")

def run_experiments(zip_path, extract_dir, depths, ensemble_sizes, timeouts, n_trials, out_csv, out_json):
    _, subsets = prepare_ensemble(zip_path, extract_dir, max(ensemble_sizes + [50]))
    first_network = subsets[max(subsets.keys())][0]
    all_moves = build_all_moves(first_network, make_sim_args(max(subsets.keys())))
    ec_map, ec_fast = make_evaluators(subsets)

    # algos = ["NMCS", "LNMCS", "NRPA", "GNRPA"]
    # algos = ["NMCS"]
    algos = ["NMCS", "LNMCS", "NRPA"]
    # algos = ["NRPA"]
    total = len(algos) * len(depths) * len(ensemble_sizes) * len(timeouts) * n_trials
    pbar = tqdm(total=total, desc="Experiment Progress", unit="run")

    results, trials = [], defaultdict(list)

    for timeout in timeouts:
        for size in ensemble_sizes:
            if size not in ec_map:
                print(f"[WARN] Ensemble size {size} not available. Skipping.")
                continue
            ec = ec_map[size]
            for depth in depths:
                for algo in algos:
                    scores, times, best_score, best_set = [], [], -np.inf, []
                    for trial in range(n_trials):
                        t0 = time.time()
                        score, muts = run_single(algo, depth, timeout, all_moves, ec, ec_fast)
                        dt = time.time() - t0
                        scores.append(score)
                        times.append(dt)
                        if score > best_score:
                            best_score, best_set = score, muts
                        trials["trials"].append({"algorithm": algo, "ensemble_size": size, "depth": depth, "timeout": timeout, "trial": trial, "score": score, "time": dt, "best_set_so_far": best_set})
                        pbar.update(1)
                    results.append({
                        "algorithm": algo,
                        "ensemble_size": size,
                        "depth": depth,
                        "timeout": timeout,
                        "mean_score": float(np.mean(scores)),
                        "std_score": float(np.std(scores)),
                        "mean_time": float(np.mean(times)),
                        "best_score": float(best_score),
                        "best_set": best_set
                    })
    pbar.close()
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")
    with open(out_json, "w") as f:
        json.dump({"summary": results, "trials": trials["trials"]}, f, indent=2)
    print(f"📝 Raw trials saved to {out_json}")


# =====================
# Entry Point
# =====================

def parse_args():
    p = argparse.ArgumentParser(description="Anytime quality experiment runner")
    p.add_argument("--zip_path", type=str, default="data/bundle-exactpkn32-nocyclic-globalfps.zip")
    p.add_argument("--extract_dir", type=str, default="data/WT_ensemble")
    p.add_argument("--depths", type=int, nargs="+", default=list(range(3, 11)))
    p.add_argument("--ensemble_sizes", type=int, nargs="+", default=[200, 500, 700, 1000])
    p.add_argument("--timeouts", type=int, nargs="+", default=[5])
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--out_csv", type=str, default="anytime_results.csv")
    p.add_argument("--out_json", type=str, default="anytime_results_trials.json")
    return p.parse_args()

def main():
    args = parse_args()
    run_experiments(
        zip_path=args.zip_path,
        extract_dir=args.extract_dir,
        depths=args.depths,
        ensemble_sizes=args.ensemble_sizes,
        timeouts=args.timeouts,
        n_trials=args.n_trials,
        out_csv=args.out_csv,
        out_json=args.out_json
    )

if __name__ == "__main__":
    main()