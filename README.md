# Mutation Set Search in Boolean Network Ensembles

This repo benchmarks nested search algorithms (NMCS, LNMCS, NRPA, GNRPA) for identifying mutation sets that achieve the good phenotype probabilities(>0.8) in ensembles of asynchronous Boolean networks.

---

## Installation

```bash
git clone https://github.com/xiczhang/Monte-Carlo-Search-on-Boolean-Networks.git 
pip install -r requirements.txt
```

## Quick Start
 Run all search algorithms (quick test)
 ```
python -m experiments.experiment \
  --zip_path data/bundle-exactpkn32-nocyclic-globalfps.zip \
  --extract_dir data/tmp_ensemble \
  --depths 7 \
  --ensemble_sizes 50 \
  --timeouts 10 \
  --n_trials 1 \
  --out_csv results/small_test.csv \
  --out_json results/small_test.json
```
📝 Runs all 4 nested search algorithms on a small ensemble in ~seconds.

## See the uniform simulation bound 
```
python -m src.plot
```

## Project Structure
```
Search4MutationSets-BN-Ensembles/
├── src/            # Core simulation + algorithms
├── experiments/    # CLI runner + argument parsing
├── data/           # Boolean networks and bundles
├── results/        # main results
├── notebooks/      # coming soon
└── README.md
```

## Models & Data
*data/bundle-exactpkn32-nocyclic-globalfps.zip* contains the ensemble used in quick tests AND *.bnet* files are extracted automatically on run

## Methods Included
> **NMCS** — Nested Monte Carlo Search  
> **LNMCS** — Lazy NMCS

> **NRPA** — Policy adaptation nested rollout  
> **GNRPA** — NRPA with guidance/bias terms
 
## 📍 
This work was conducted at LAMSADE, Université Paris-Dauphine. A detailed manuscript and results is available upon request.



