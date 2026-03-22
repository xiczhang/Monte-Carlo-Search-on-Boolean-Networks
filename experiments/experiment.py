#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark runner for Boolean-network mutation-set search algorithms
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import random
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure local imports work when running from the repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import src.bn_async_sim as bnas
import src.ensemble_utils as eu
from src.bilnmcs_module import bilnmcs
from src.lnmcs_module import lnmcs
from src.nmcs_module import nmcs
from src.nrpa_module import nrpa


Mutation = Tuple[str, bool]
MutationSet = List[Mutation]


# =====================
# Reproducibility helpers
# =====================


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


# =====================
# Simulation / evaluation utilities
# =====================


def adaptive_sims_per_model(ensemble_size: int, total_budget: int = 2500) -> int:
    """Keep total simulated trajectories roughly constant across ensemble sizes."""
    return max(1, int(math.ceil(total_budget / max(1, ensemble_size))))


DEFAULT_OUTPUT_NODES = ["Apoptosis", "CellCycleArrest", "Invasion", "EMT"]
DEFAULT_TARGET_PHENO = ["CellCycleArrest", "Invasion", "EMT"]
DEFAULT_INITIAL_ACTIVATION = {
    "miR200": 1,
    "miR203": 1,
    "miR34": 1,
    "DNAdamage": 0.5,
    "ECMicroenv": 0.5,
}


def make_sim_args(
    ensemble_size: int,
    *,
    output_nodes: Sequence[str] = DEFAULT_OUTPUT_NODES,
    target_pheno: Sequence[str] = DEFAULT_TARGET_PHENO,
    sims_per_model: Optional[int] = None,
    total_budget: int = 2500,
    max_steps: int = 500,
    initial_activation: Optional[Dict[str, float]] = None,
    sample_interval: int = 1,
    threshold: float = 0.0,
) -> Dict[str, object]:
    return {
        "output_nodes": list(output_nodes),
        "sims_per_model": sims_per_model if sims_per_model is not None else adaptive_sims_per_model(ensemble_size, total_budget=total_budget),
        "max_steps": max_steps,
        "initial_activation": dict(DEFAULT_INITIAL_ACTIVATION if initial_activation is None else initial_activation),
        "sample_interval": sample_interval,
        "threshold": threshold,
        "target_pheno": tuple(target_pheno),
    }


class EvalCounter:
    """Evaluator with cache and budget accounting.

    Notes:
    - `queries` counts all calls from the search algorithm.
    - `cache_misses` counts actual expensive evaluations.
    - `simulated_trajectories` is an approximate but useful cost proxy.
    """

    def __init__(self, networks: Sequence[bnas.BooleanNetwork], sim_args: Dict[str, object]):
        self.networks = list(networks)
        self.sim_args = dict(sim_args)
        self.queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache: Dict[Tuple[Mutation, ...], float] = {}

    def evaluate(self, muts: Sequence[Mutation]) -> float:
        self.queries += 1
        key = normalize_key(muts)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        mutated = [bnas.create_mutant_network(net, list(muts)) for net in self.networks]
        hist = eu.build_ensemble_history(
            mutated,
            self.sim_args["output_nodes"],
            self.sim_args["sims_per_model"],
            self.sim_args["max_steps"],
            self.sim_args["initial_activation"],
            self.sim_args["sample_interval"],
            threshold=self.sim_args["threshold"],
            verbose=False,
        )
        score = hist[-1][1].get(tuple(self.sim_args["target_pheno"]), 0.0)
        self._cache[key] = score
        return score

    @property
    def simulated_trajectories(self) -> int:
        return int(len(self.networks) * int(self.sim_args["sims_per_model"]) * self.cache_misses)

    def stats(self) -> Dict[str, int]:
        return {
            "queries": self.queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "simulated_trajectories": self.simulated_trajectories,
            "ensemble_size": len(self.networks),
            "sims_per_model": int(self.sim_args["sims_per_model"]),
        }


# =====================
# Mutation-set helpers
# =====================


def normalize_sorted_list(state_list: Sequence[Mutation]) -> MutationSet:
    return sorted(list(state_list), key=lambda x: (x[0], x[1]))


def normalize_key(state_list: Sequence[Mutation]) -> Tuple[Mutation, ...]:
    return tuple(normalize_sorted_list(state_list))


def canonicalize_mutation_set(muts: Optional[Sequence[Sequence[object]]]) -> MutationSet:
    """Return a valid unique-gene mutation set.

    If a gene appears multiple times, the last assignment wins. This matches the
    semantics of `dict(mutations)` used in `create_mutant_network`, while also
    producing a valid explicit mutation set for fair rescoring.
    """
    if not muts:
        return []

    order: List[str] = []
    values: Dict[str, bool] = {}
    for item in muts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        gene = str(item[0])
        value = bool(item[1])
        if gene not in values:
            order.append(gene)
        values[gene] = value
    out = [(gene, values[gene]) for gene in order]
    return normalize_sorted_list(out)


# =====================
# Data loading
# =====================


def _guess_defaults() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return `(ensemble_dir, zip_path, extract_dir)` guesses if possible."""
    candidates = [
        SCRIPT_DIR / "data" / "WT_ensemble",
        SCRIPT_DIR.parent / "data" / "WT_ensemble",
        Path("/mnt/data/_code_extract/Search4MutationSets-BN-Ensembles-main/data/WT_ensemble"),
        Path("/mnt/data/Search4MutationSets-BN-Ensembles-main/data/WT_ensemble"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return str(p), None, None

    zip_candidates = [
        SCRIPT_DIR / "code.zip",
        SCRIPT_DIR / "data" / "bundle-exactpkn32-nocyclic-globalfps.zip",
        SCRIPT_DIR.parent / "data" / "bundle-exactpkn32-nocyclic-globalfps.zip",
        Path("/mnt/data/code.zip"),
    ]
    for z in zip_candidates:
        if z.exists() and z.is_file():
            extract_dir = str(SCRIPT_DIR / "_ensemble_extract")
            return None, str(z), extract_dir

    return None, None, None


def _find_best_bnet_dir(root_dir: str) -> Tuple[str, List[str]]:
    """Choose the directory under `root_dir` that contains the most .bnet files."""
    pattern = os.path.join(root_dir, "**", "*.bnet")
    all_files = glob.glob(pattern, recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No .bnet files found under {root_dir}")

    by_dir: Dict[str, List[str]] = defaultdict(list)
    for fp in all_files:
        by_dir[os.path.dirname(fp)].append(fp)

    best_dir, best_files = max(by_dir.items(), key=lambda kv: len(kv[1]))
    best_files = sorted(best_files)
    return best_dir, best_files



def resolve_ensemble_dir(
    ensemble_dir: Optional[str],
    zip_path: Optional[str],
    extract_dir: Optional[str],
) -> Tuple[str, List[str]]:
    if ensemble_dir:
        best_dir, files = _find_best_bnet_dir(ensemble_dir)
        return best_dir, files

    if zip_path:
        if not extract_dir:
            extract_dir = str(SCRIPT_DIR / "_ensemble_extract")
        os.makedirs(extract_dir, exist_ok=True)
        if not os.listdir(extract_dir):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        best_dir, files = _find_best_bnet_dir(extract_dir)
        return best_dir, files

    guessed_ensemble_dir, guessed_zip_path, guessed_extract_dir = _guess_defaults()
    if guessed_ensemble_dir:
        best_dir, files = _find_best_bnet_dir(guessed_ensemble_dir)
        return best_dir, files
    if guessed_zip_path:
        return resolve_ensemble_dir(None, guessed_zip_path, guessed_extract_dir)

    raise ValueError("Provide --ensemble_dir or --zip_path (no dataset source could be auto-detected).")



def prepare_ensemble(
    ensemble_dir: Optional[str],
    zip_path: Optional[str],
    extract_dir: Optional[str],
    required_sizes: Sequence[int],
    network_shuffle_seed: int,
) -> Tuple[str, List[bnas.BooleanNetwork], Dict[int, List[bnas.BooleanNetwork]]]:
    resolved_dir, model_files = resolve_ensemble_dir(ensemble_dir, zip_path, extract_dir)
    networks = [bnas.load_network_model(f) for f in sorted(model_files)]
    rnd = random.Random(network_shuffle_seed)
    rnd.shuffle(networks)

    available_n = len(networks)
    subsets: Dict[int, List[bnas.BooleanNetwork]] = {}
    for size in sorted(set(int(s) for s in required_sizes)):
        if size <= available_n:
            subsets[size] = networks[:size]
    return resolved_dir, networks, subsets



def build_all_moves(network: bnas.BooleanNetwork, output_nodes: Sequence[str]) -> List[Mutation]:
    raw = eu.generate_single_mutants(network, output_nodes)
    return [tuple(m[0]) if isinstance(m, list) else tuple(m) for m in raw]


# =====================
# Search / validation
# =====================


def make_search_evaluators(
    subsets: Dict[int, List[bnas.BooleanNetwork]],
    ensemble_size: int,
    fast_ensemble_size: int,
    main_total_budget: int,
    main_sims_per_model: Optional[int],
    fast_sims_per_model: int,
) -> Tuple[EvalCounter, EvalCounter]:
    if ensemble_size not in subsets:
        raise KeyError(f"Main ensemble size {ensemble_size} is unavailable.")

    main_networks = subsets[ensemble_size]
    fast_size = min(fast_ensemble_size, len(main_networks), max(subsets.keys()))
    # Prefer the requested fast size if available globally, else largest available <= requested.
    eligible_fast_sizes = [s for s in subsets.keys() if s <= fast_ensemble_size]
    if eligible_fast_sizes:
        chosen_fast_size = max(eligible_fast_sizes)
    else:
        chosen_fast_size = min(subsets.keys())
    fast_networks = subsets[chosen_fast_size]

    main_args = make_sim_args(
        ensemble_size,
        sims_per_model=main_sims_per_model,
        total_budget=main_total_budget,
    )
    fast_args = make_sim_args(
        len(fast_networks),
        sims_per_model=fast_sims_per_model,
        total_budget=main_total_budget,
    )
    return EvalCounter(main_networks, main_args), EvalCounter(fast_networks, fast_args)



def score_mutation_set_once(
    muts: Sequence[Mutation],
    networks: Sequence[bnas.BooleanNetwork],
    sim_args: Dict[str, object],
) -> float:
    mutated = [bnas.create_mutant_network(net, list(muts)) for net in networks]
    hist = eu.build_ensemble_history(
        mutated,
        sim_args["output_nodes"],
        sim_args["sims_per_model"],
        sim_args["max_steps"],
        sim_args["initial_activation"],
        sim_args["sample_interval"],
        threshold=sim_args["threshold"],
        verbose=False,
    )
    return hist[-1][1].get(tuple(sim_args["target_pheno"]), 0.0)



def validate_solution(
    muts: Sequence[Mutation],
    networks: Sequence[bnas.BooleanNetwork],
    *,
    ensemble_size: int,
    main_total_budget: int,
    validate_reps: int,
    validate_sims_per_model: Optional[int],
    base_seed: int,
) -> Tuple[float, float, List[float]]:
    sim_args = make_sim_args(
        ensemble_size,
        sims_per_model=validate_sims_per_model,
        total_budget=main_total_budget,
    )
    vals: List[float] = []
    for rep in range(max(1, validate_reps)):
        set_all_seeds(base_seed + rep)
        vals.append(float(score_mutation_set_once(muts, networks, sim_args)))
    return float(np.mean(vals)), float(np.std(vals)), vals



def run_single_algorithm(
    config: Dict[str, object],
    *,
    depth: int,
    timeout_sec: float,
    all_moves: Sequence[Mutation],
    ec_main: EvalCounter,
    ec_fast: EvalCounter,
) -> Tuple[float, MutationSet]:
    algo = str(config["algorithm"])
    level = int(config["level"])

    if algo == "NMCS":
        score, muts = nmcs([], level=level, depth=depth, best_moves=list(all_moves), ec=ec_main, timeout_sec=timeout_sec)
        return float(score), canonicalize_mutation_set(muts)

    if algo == "LNMCS":
        score, muts = lnmcs(
            [],
            level=level,
            depth=depth,
            all_moves=list(all_moves),
            ec=ec_main,
            b=int(config["lnmcs_b"]),
            r=float(config["lnmcs_r"]),
            e=int(config["lnmcs_e"]),
            timeout_sec=timeout_sec,
        )
        return float(score), canonicalize_mutation_set(muts)

    if algo == "BILNMCS":
        score, muts = bilnmcs(
            [],
            level=level,
            depth=depth,
            all_moves=list(all_moves),
            ec_main=ec_main,
            ec_fast=ec_fast,
            b=int(config["bilnmcs_b"]),
            r=float(config["bilnmcs_r"]),
            e=int(config["bilnmcs_e"]),
            timeout_sec=timeout_sec,
        )
        return float(score), canonicalize_mutation_set(muts)

    if algo == "NRPA":
        score, muts = nrpa(level=level, policy={}, depth=depth, ec=ec_main, all_moves=list(all_moves), timeout_sec=timeout_sec)
        return float(score), canonicalize_mutation_set(muts)

    raise ValueError(f"Unknown algorithm: {algo}")


# =====================
# Main experiment loop
# =====================


def summarize_trials(trial_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        key = (
            row["algorithm"],
            row["algorithm_config_key"],
            row["ensemble_size"],
            row["depth"],
            row["timeout_sec"],
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        rows = list(rows)
        best_row = max(rows, key=lambda r: float(r["validated_score_mean"]))
        finite_raw = [float(r["raw_search_score"]) for r in rows if np.isfinite(float(r["raw_search_score"]))]
        summary_rows.append(
            {
                "algorithm": key[0],
                "algorithm_config": rows[0]["algorithm_config"],
                **flatten_algorithm_config(rows[0]["algorithm_config"]),
                "ensemble_size": key[2],
                "depth": key[3],
                "timeout_sec": key[4],
                "n_trials": len(rows),
                "mean_validated_score": float(np.mean([r["validated_score_mean"] for r in rows])),
                "std_validated_score": float(np.std([r["validated_score_mean"] for r in rows])),
                "mean_raw_search_score": float(np.mean(finite_raw)) if finite_raw else float("nan"),
                "mean_elapsed_sec": float(np.mean([r["elapsed_sec"] for r in rows])),
                "mean_returned_size": float(np.mean([r["returned_size"] for r in rows])),
                "mean_total_simulated_trajectories": float(np.mean([r["total_simulated_trajectories"] for r in rows])),
                "best_validated_score": float(best_row["validated_score_mean"]),
                "best_mutation_set": best_row["returned_set"],
                "best_trial": int(best_row["trial"]),
            }
        )
    return summary_rows


def _resolve_values(single_value: object, sweep_values: Optional[Sequence[object]]) -> List[object]:
    if sweep_values:
        return list(sweep_values)
    return [single_value]


def flatten_algorithm_config(config: Dict[str, object]) -> Dict[str, object]:
    return {f"algo_{key}": value for key, value in config.items() if key != "algorithm"}


def build_algorithm_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    levels = [int(v) for v in _resolve_values(args.level, args.levels)]
    configs: List[Dict[str, object]] = []

    for algo in args.algorithms:
        if algo == "NMCS":
            for level in levels:
                configs.append({"algorithm": algo, "level": level})
            continue

        if algo == "LNMCS":
            for level, b, r, e in itertools.product(
                levels,
                _resolve_values(args.lnmcs_b, args.lnmcs_b_values),
                _resolve_values(args.lnmcs_r, args.lnmcs_r_values),
                _resolve_values(args.lnmcs_e, args.lnmcs_e_values),
            ):
                configs.append({
                    "algorithm": algo,
                    "level": int(level),
                    "lnmcs_b": int(b),
                    "lnmcs_r": float(r),
                    "lnmcs_e": int(e),
                })
            continue

        if algo == "BILNMCS":
            for level, b, r, e in itertools.product(
                levels,
                _resolve_values(args.bilnmcs_b, args.bilnmcs_b_values),
                _resolve_values(args.bilnmcs_r, args.bilnmcs_r_values),
                _resolve_values(args.bilnmcs_e, args.bilnmcs_e_values),
            ):
                configs.append({
                    "algorithm": algo,
                    "level": int(level),
                    "bilnmcs_b": int(b),
                    "bilnmcs_r": float(r),
                    "bilnmcs_e": int(e),
                })
            continue

        if algo == "NRPA":
            for level in levels:
                configs.append({"algorithm": algo, "level": level})
            continue

        raise ValueError(f"Unsupported algorithm configuration: {algo}")

    return configs



def run_experiments(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    requested_sizes = sorted(set(int(s) for s in args.ensemble_sizes + [args.fast_ensemble_size]))
    resolved_dir, all_networks, subsets = prepare_ensemble(
        args.ensemble_dir,
        args.zip_path,
        args.extract_dir,
        requested_sizes,
        args.network_shuffle_seed,
    )

    if not subsets:
        raise RuntimeError("No requested ensemble size is available from the supplied dataset.")

    available_sizes = sorted(subsets.keys())
    run_sizes = [s for s in args.ensemble_sizes if s in subsets]
    skipped_sizes = [s for s in args.ensemble_sizes if s not in subsets]

    first_network = subsets[max(available_sizes)][0]
    all_moves = build_all_moves(first_network, DEFAULT_OUTPUT_NODES)

    algo_configs = build_algorithm_configs(args)
    total_runs = len(algo_configs) * len(run_sizes) * len(args.depths) * len(args.timeouts) * args.n_trials
    progress = tqdm(total=total_runs, desc="Benchmark", unit="run")

    trials: List[Dict[str, object]] = []

    for timeout_sec in args.timeouts:
        for ensemble_size in run_sizes:
            for depth in args.depths:
                for algo_config in algo_configs:
                    for trial in range(args.n_trials):
                        search_seed = int(args.seed_base + 100000 * trial + 1000 * depth + 10 * ensemble_size + int(round(timeout_sec * 100)))
                        val_seed = search_seed + 10_000_000

                        set_all_seeds(search_seed)
                        ec_main, ec_fast = make_search_evaluators(
                            subsets,
                            ensemble_size,
                            args.fast_ensemble_size,
                            args.main_total_budget,
                            args.main_sims_per_model,
                            args.fast_sims_per_model,
                        )

                        error_msg = None
                        raw_score = float("-inf")
                        returned_set: MutationSet = []
                        started = time.perf_counter()
                        try:
                            raw_score, returned_set = run_single_algorithm(
                                algo_config,
                                depth=depth,
                                timeout_sec=float(timeout_sec),
                                all_moves=all_moves,
                                ec_main=ec_main,
                                ec_fast=ec_fast,
                            )
                        except Exception as exc:  # pragma: no cover - defensive benchmarking path
                            error_msg = f"{type(exc).__name__}: {exc}"
                            returned_set = []
                            raw_score = float("-inf")
                        elapsed = time.perf_counter() - started

                        if not np.isfinite(raw_score):
                            raw_score = float("nan")
                        returned_set = canonicalize_mutation_set(returned_set)
                        validated_mean, validated_std, _ = validate_solution(
                            returned_set,
                            subsets[ensemble_size],
                            ensemble_size=ensemble_size,
                            main_total_budget=args.main_total_budget,
                            validate_reps=args.validate_reps,
                            validate_sims_per_model=args.validate_sims_per_model,
                            base_seed=val_seed,
                        )

                        main_stats = ec_main.stats()
                        fast_stats = ec_fast.stats()
                        trials.append(
                            {
                                "algorithm": algo_config["algorithm"],
                                "algorithm_config": dict(algo_config),
                                "algorithm_config_key": json.dumps(algo_config, sort_keys=True),
                                **flatten_algorithm_config(algo_config),
                                "ensemble_size": ensemble_size,
                                "depth": depth,
                                "timeout_sec": float(timeout_sec),
                                "trial": trial,
                                "elapsed_sec": float(elapsed),
                                "raw_search_score": float(raw_score),
                                "validated_score_mean": validated_mean,
                                "validated_score_std": validated_std,
                                "returned_size": len(returned_set),
                                "returned_set": returned_set,
                                "total_simulated_trajectories": main_stats["simulated_trajectories"] + fast_stats["simulated_trajectories"],
                                "error": error_msg,
                            }
                        )
                        progress.update(1)

    progress.close()
    summary = summarize_trials(trials)
    meta = {
        "resolved_ensemble_dir": resolved_dir,
        "n_available_networks": len(all_networks),
        "available_sizes": available_sizes,
        "requested_sizes": list(args.ensemble_sizes),
        "skipped_sizes": skipped_sizes,
        "n_all_moves": len(all_moves),
        "algorithms": list(args.algorithms),
        "algorithm_configs": algo_configs,
        "main_total_budget": args.main_total_budget,
        "main_sims_per_model": args.main_sims_per_model,
        "fast_ensemble_size": args.fast_ensemble_size,
        "fast_sims_per_model": args.fast_sims_per_model,
        "validate_reps": args.validate_reps,
        "validate_sims_per_model": args.validate_sims_per_model,
        "network_shuffle_seed": args.network_shuffle_seed,
        "seed_base": args.seed_base,
    }
    return summary, trials, meta


# =====================
# CLI
# =====================


def parse_args() -> argparse.Namespace:
    guessed_ensemble_dir, guessed_zip_path, guessed_extract_dir = _guess_defaults()

    p = argparse.ArgumentParser(description="Fair benchmark runner")

    # Dataset source
    p.add_argument("--ensemble_dir", type=str, default=guessed_ensemble_dir)
    p.add_argument("--zip_path", type=str, default=guessed_zip_path)
    p.add_argument("--extract_dir", type=str, default=guessed_extract_dir)

    # Sweep
    p.add_argument("--depths", type=int, nargs="+", default=[6])
    p.add_argument("--ensemble_sizes", type=int, nargs="+", default=[50, 200])
    p.add_argument("--timeouts", type=float, nargs="+", default=[5.0])
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["NMCS", "LNMCS", "BILNMCS", "NRPA"],
        choices=["NMCS", "LNMCS", "BILNMCS", "NRPA"],
    )
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--levels", type=int, nargs="+", default=None, help="Optional sweep over search levels; defaults to --level.")

    # Evaluator configuration
    p.add_argument("--main_total_budget", type=int, default=2500, help="Adaptive total simulation budget for the main evaluator if --main_sims_per_model is not set.")
    p.add_argument("--main_sims_per_model", type=int, default=None, help="Override adaptive main sims/model with a fixed value.")
    p.add_argument("--fast_ensemble_size", type=int, default=50)
    p.add_argument("--fast_sims_per_model", type=int, default=5, help="Keep the fast evaluator truly cheap; 5 on 50 models is 250 trajectories.")
    p.add_argument("--validate_reps", type=int, default=3, help="How many fresh rescoring reps to average for the reported benchmark score.")
    p.add_argument("--validate_sims_per_model", type=int, default=None, help="Override validation sims/model; defaults to the main evaluator setting.")

    # Common lazy search params
    p.add_argument("--lnmcs_b", type=int, default=3)
    p.add_argument("--lnmcs_r", type=float, default=0.4)
    p.add_argument("--lnmcs_e", type=int, default=10)
    p.add_argument("--bilnmcs_b", type=int, default=3)
    p.add_argument("--bilnmcs_r", type=float, default=0.4)
    p.add_argument("--bilnmcs_e", type=int, default=10)
    p.add_argument("--lnmcs_b_values", type=int, nargs="+", default=None, help="Optional LNMCS sweep values; defaults to --lnmcs_b.")
    p.add_argument("--lnmcs_r_values", type=float, nargs="+", default=None, help="Optional LNMCS sweep values; defaults to --lnmcs_r.")
    p.add_argument("--lnmcs_e_values", type=int, nargs="+", default=None, help="Optional LNMCS sweep values; defaults to --lnmcs_e.")
    p.add_argument("--bilnmcs_b_values", type=int, nargs="+", default=None, help="Optional BILNMCS sweep values; defaults to --bilnmcs_b.")
    p.add_argument("--bilnmcs_r_values", type=float, nargs="+", default=None, help="Optional BILNMCS sweep values; defaults to --bilnmcs_r.")
    p.add_argument("--bilnmcs_e_values", type=int, nargs="+", default=None, help="Optional BILNMCS sweep values; defaults to --bilnmcs_e.")

    # Reproducibility / output
    p.add_argument("--seed_base", type=int, default=12345)
    p.add_argument("--network_shuffle_seed", type=int, default=2)
    p.add_argument("--out_csv", type=str, default="benchmark_summary.csv")
    p.add_argument("--out_json", type=str, default="benchmark_trials.json")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    summary_rows, trial_rows, meta = run_experiments(args)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary_rows, "trials": trial_rows}, f, indent=2)

    print(f"\nSaved summary CSV to {out_csv}")
    print(f"Saved trial JSON to {out_json}")
    print(f"Resolved ensemble dir: {meta['resolved_ensemble_dir']}")
    if meta["skipped_sizes"]:
        print(f"Skipped unavailable ensemble sizes: {meta['skipped_sizes']}")


if __name__ == "__main__":
    main()
