import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nearest_available(values, target):
    values = np.array(sorted(values))
    return int(values[np.argmin(np.abs(values - target))])

def parse_list_arg(s):
    if s is None or not s.strip():
        return []
    return [int(x) for x in s.replace(",", " ").split() if x.strip()]

def map_to_available_list(requested, pool):
    pool_sorted = sorted(pool)
    mapped = []
    for v in requested:
        mapped.append(nearest_available(pool_sorted, v))
    # deduplicate but keep order
    seen = set(); keep = []
    for v in mapped:
        if v not in seen:
            keep.append(v); seen.add(v)
    return keep

def main():
    ap = argparse.ArgumentParser(description="Simple plots with user-set bound lists.")
    ap.add_argument("--csv", default="results/vb.csv", help="Path to vb_wc_only.csv")
    ap.add_argument("--outdir", default="results/figs", help="Output folder")
    ap.add_argument("--emp_n", type=int, default=200, help="n for empirical curve in vs-M plot")
    ap.add_argument("--emp_M", type=int, default=200, help="M for empirical curve in vs-n plot")
    ap.add_argument("--bounds_n", default="10,50,200", help="n values for worst-case curves in vs-M plot")
    ap.add_argument("--bounds_M", default="20,200,700", help="M values for worst-case curves in vs-n plot")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    for col in ["M","n","V_emp","V_wc"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    all_Ms = sorted(df["M"].unique())
    all_ns = sorted(df["n"].unique())

    n_emp = nearest_available(all_ns, args.emp_n)
    M_emp = nearest_available(all_Ms, args.emp_M)

    req_n_bounds = parse_list_arg(args.bounds_n)
    req_M_bounds = parse_list_arg(args.bounds_M)
    n_bounds = map_to_available_list(req_n_bounds, all_ns)
    M_bounds = map_to_available_list(req_M_bounds, all_Ms)

    if n_emp not in n_bounds: n_bounds.append(n_emp)
    if M_emp not in M_bounds: M_bounds.append(M_emp)

    n_bounds = sorted(set(n_bounds))
    M_bounds = sorted(set(M_bounds))

    print(f"Empirical curves: n={n_emp} (vs M), M={M_emp} (vs n)")
    print(f"Bound n values (plot vs M): {n_bounds}")
    print(f"Bound M values (plot vs n): {M_bounds}")

    # ---------- Plot 1: V vs M (empirical uses n = n_emp) ----------
    sub_emp = df[df["n"] == n_emp].sort_values("M")
    if sub_emp.empty:
        raise ValueError(f"No rows for empirical n={n_emp}")

    plt.figure(figsize=(8,5))
    # empirical first so it appears first in legend
    plt.plot(sub_emp["M"], sub_emp["V_emp"], marker="o", linestyle="-", linewidth=2.0,
             label=f"Empirical (n={n_emp})")

    for n in n_bounds:
        sub_b = df[df["n"] == n].sort_values("M")
        if sub_b.empty: 
            continue
        plt.plot(sub_b["M"], sub_b["V_wc"], marker="s", linestyle="--", linewidth=1.8,
                 label=f"Worst-case (n={n})")

    plt.xlabel("Ensemble size M")
    plt.ylabel("Variance of $\hat{p}_{ens}$")
    plt.title("Variance vs M ")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"variance_vs_M_emp_n{n_emp}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- Plot 2: V vs n (empirical uses M = M_emp) ----------
    sub_emp2 = df[df["M"] == M_emp].sort_values("n")
    if sub_emp2.empty:
        raise ValueError(f"No rows for empirical M={M_emp}")

    plt.figure(figsize=(8,5))
    plt.plot(sub_emp2["n"], sub_emp2["V_emp"], marker="o", linestyle="-", linewidth=2.0,
             label=f"Empirical (M={M_emp})")

    for M in M_bounds:
        sub_b2 = df[df["M"] == M].sort_values("n")
        if sub_b2.empty:
            continue
        plt.plot(sub_b2["n"], sub_b2["V_wc"], marker="s", linestyle="--", linewidth=1.8,
                 label=f"Worst-case (M={M})")

    plt.xlabel("Trajectories per model $n_{traj}$")
    plt.ylabel("Variance of $\hat{p}_{ens}$")
    plt.title("Variance vs $n_{traj}$")
    plt.grid(True, alpha=0.4)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"variance_vs_n_emp_M{M_emp}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("Done. Figures saved in:", args.outdir)

if __name__ == "__main__":
    main()
