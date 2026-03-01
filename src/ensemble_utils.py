import numpy as np
from tqdm import tqdm
import bn_async_sim as bnas
import pandas as pd


from tqdm import tqdm  # ensure tqdm is imported

def build_ensemble_history(
    networks,
    output_nodes,
    sims_per_model,
    max_steps,
    initial_activation,
    sample_interval,
    threshold=0.001,
    verbose=True
):
    """
    1) Runs convergence on each BooleanNetwork in `networks`.
    2) Aggregates phenotype probabilities over time,
       drops any phenotype below `threshold`, and renormalizes.

    Parameters
    ----------
    networks : List[BooleanNetwork]
    output_nodes : List[str]
    sims_per_model : int
    max_steps : int
    initial_activation : Dict[str, float]
    sample_interval : int
    threshold : float, optional
    verbose : bool, optional
        If False, skips the tqdm bar and summary print.
    """
    # STEP 1: simulate each model and collect histories
    all_histories = []
    iterator = tqdm(networks, desc="Models") if verbose else networks

    for net in iterator:
        history, _ = bnas.track_convergence(
            net,
            output_nodes,
            sims_per_model,
            max_steps,
            initial_activation,
            sample_interval
        )
        all_histories.append(history)

    if verbose:
        print(f"{len(all_histories)} histories collected for models.")

    # assume all histories share same timepoints
    timepoints = [t for t, _ in all_histories[0]]
    ensemble_history = []

    # STEP 2: aggregate across models at each timepoint
    for idx, t in enumerate(timepoints):
        per_model = [hist[idx][1] for hist in all_histories]

        # union of all phenotypes
        all_phenos = set().union(*(pd.keys() for pd in per_model))

        # compute mean probability for each phenotype
        mean_probs = {
            pheno: sum(pd.get(pheno, 0.0) for pd in per_model) / len(per_model)
            for pheno in all_phenos
        }

        # drop negligible phenotypes and renormalize
        filtered = {ph: p for ph, p in mean_probs.items() if p >= threshold}
        total = sum(filtered.values())
        if total > 0:
            for ph in filtered:
                filtered[ph] /= total

        ensemble_history.append((t, filtered))

    return ensemble_history


def show_ensemble_results(ensemble_history, tag="WT"):
    """
    Print the final steady-state phenotype probabilities for an ensemble
    and then plot its convergence curve.

    Parameters
    ----------
    ensemble_history : List[(step, {phenotype: prob})]
        The output of build_ensemble_history or track_convergence.
    tag : str, optional
        A label to include in the heading (e.g. 'WT', 'Mutant').
    """
    # Extract final probabilities
    final_probs = ensemble_history[-1][1]

    print(f"Final ensemble ({tag}) phenotype probabilities:")
    for pheno in sorted(final_probs):
        prob = final_probs[pheno]
        if isinstance(pheno, (list, tuple)):
            label = ", ".join(pheno)
        else:
            label = str(pheno)
        print(f"  {label:30s} {prob:.4f}")
    # Plot convergence
    bnas.plot_convergence(ensemble_history, mutations=None)
    
def generate_single_mutants(network, target_markers):
    """
    Generate all single-gene mutants (ON and OFF) excluding `target_markers`.

    Parameters
    ----------
    network : BooleanNetwork
    target_markers : set[str]
        Genes to exclude from mutation.

    Returns
    -------
    List of single-gene mutant specs.
    """
    muts = []
    for g in network.node_names:
        if g in target_markers:
            continue
        muts.append([(g, True)])
        muts.append([(g, False)])
    return muts


def generate_double_mutants(network, target_markers):
    """
    Generate all double-gene mutants (combinations of ON/OFF) excluding `target_markers`.

    Parameters
    ----------
    network : BooleanNetwork
    target_markers : set[str]
        Genes to exclude from mutation.

    Returns
    -------
    List of double-gene mutant specs.
    """
    muts = []
    genes = [g for g in network.node_names if g not in target_markers]
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            g1, g2 = genes[i], genes[j]
            for v1 in (False, True):
                for v2 in (False, True):
                    muts.append([(g1, v1), (g2, v2)])
    return muts


def generate_triple_mutants(network, target_markers):
    """
    Generate all triple-gene mutants (combinations of ON/OFF) excluding `target_markers`.

    Parameters
    ----------
    network : BooleanNetwork
    target_markers : set[str]
        Genes to exclude from mutation.

    Returns
    -------
    List of triple-gene mutant specs.
    """
    muts = []
    genes = [g for g in network.node_names if g not in target_markers]
    n = len(genes)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                g1, g2, g3 = genes[i], genes[j], genes[k]
                for v1 in (False, True):
                    for v2 in (False, True):
                        for v3 in (False, True):
                            muts.append([(g1, v1), (g2, v2), (g3, v3)])
    return muts

def screen_mutants(
    networks,
    mutants,
    output_nodes,
    sims_per_model,
    max_steps,
    initial_activation,
    sample_interval,
    target
):
    """
    Screen a list of mutants on an ensemble, compute score for `target`,
    save results to CSV and return a sorted DataFrame.

    Each mutant is a list of (gene, value) pairs. The score is the final
    frequency of `target` phenotype in the ensemble.
    """
    results=[]
    for mut in tqdm(mutants, desc="Screening mutants"):
        mutated=[ bnas.create_mutant_network(net, mut) for net in networks ]
        hist=build_ensemble_history(
            mutated, output_nodes, sims_per_model,
            max_steps, initial_activation, sample_interval
        )
        final_dist=hist[-1][1]
        score=final_dist.get(target,0.0)
        results.append({"mutant":mut, f"p_{target}":score})
        
    return results

def evaluate_mutant_ensemble(
    mutation,
    networks,
    output_nodes,
    init_probs,
    target_phenotype,
    sims_per_model,
    max_steps
):
    """
    Returns the fraction of (network, simulation) trials that hit exactly `target_phenotype`
    for a given mutation across the entire ensemble.

    Parameters
    ----------
    mutation : List[(str, bool)]
        The perturbation (e.g., single‐gene knockout or forced‐ON) as a list of (gene, value) pairs.
    networks : List[BooleanNetwork]
        The ensemble of Boolean‐network variants to test.
    output_nodes : List[str]
        Nodes whose values define the phenotype.
    init_probs : Dict[str, float]
        Activation probabilities for constructing random initial states.
    target_phenotype : Tuple[str, ...]
        The exact combination of output‐nodes that counts as a “hit.”
    sims_per_model : int
        Number of asynchronous sim to run per network in the ensemble.
    max_steps : int
        Max asynchronous update steps per sim.

    Returns
    -------
    float
        Fraction of total (network × trajectory) pairs that reach `target_phenotype`.
    """
    total_hits = 0
    total_trials = sims_per_model * len(networks)

    for net in networks:
        # Create the mutated network for this variant
        mutated = bnas.create_mutant_network(net, mutation)
        out_idx = [mutated.node_index[n] for n in output_nodes] # Indices of output nodes 
        # so we can quickly extract them from the simulation result

        # Run sims_per_model independent asynchronous simulations
        for _ in range(sims_per_model):
            st0 = bnas.create_initial_state(mutated, init_probs)
            out_bool, _ = bnas.run_simulation(mutated, st0, out_idx, max_steps)
            # Determine which outputs are “ON”
            ph = tuple(n for n, flag in zip(output_nodes, out_bool) if flag)
            if ph == target_phenotype:
                total_hits += 1

    return total_hits / total_trials