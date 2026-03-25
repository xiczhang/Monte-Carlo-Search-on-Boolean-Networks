# Mutation Set Search in Boolean Network Ensembles

This repo benchmarks nested search algorithms (NMCS, LNMCS, NRPA, GNRPA, BILNMCS) for identifying mutation sets that achieve the good phenotype probabilities(>0.8) in ensembles of asynchronous Boolean networks.

---

## Installation

```bash
git clone https://github.com/xiczhang/Monte-Carlo-Search-on-Boolean-Networks.git 
pip install -r requirements.txt
```

## Quick Start and Reproducing 
 try BILNMCS algorithm (quick test/ex.Scalability)
 ```
python3 experiments/experiment.py \
  --algorithms BILNMCS \
  --depths 7 \
  --ensemble_sizes 1000 \
  --timeouts 60 \
  --n_trials 10 \
  --levels 2 \
```
📝 Runs **Bi-Lazy Nested Search** (BILNMCS) algorithms on a big ensemble

To reproduce ex.Anytime
 ```
python3 experiments/experiment.py \
  --algorithms NMCS LNMCS BILNMCS NRPA \
  --depths 6 8 9 10 11 12 13 \
  --ensemble_sizes 1000 \
  --timeouts 10 20 30 \
  --n_trials 10 \

```
📝 See the results/ex_

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
>> **NMCS** — Nested Monte Carlo Search  
>> **LNMCS** — Lazy NMCS
>
>> **BILNMCS** — Bi-Lazy NMCS
> 
>> **NRPA** — Policy adaptation nested rollout  
>> **GNRPA** — NRPA with guidance/bias terms
 
 
## 📍 
This work was conducted at *LAMSADE, Université Paris Dauphine - PSL*. A detailed manuscript and results is available upon request.



