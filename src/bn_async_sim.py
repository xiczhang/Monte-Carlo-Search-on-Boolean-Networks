import numpy as np
import random
import re
import time
import matplotlib.pyplot as plt


class BooleanNetwork:
    """
    Represents a Boolean Network with asynchronous updates.

    Attributes:
        node_names: List[str] of node identifiers.
        node_index: Dict[str, int] mapping node names to indices.
        update_funcs: List[Callable[[List[bool]], bool]] update function per node.
        influence_map: List[List[int]] mapping each node to nodes it influences.
    """
    def __init__(self, node_names, node_index, update_funcs, influence_map):
        self.node_names = node_names
        self.node_index = node_index
        self.update_funcs = update_funcs
        self.influence_map = influence_map


def load_network_model(filename, separator=","):
    """
    Load a network model from a .bnet file into a BooleanNetwork.

    File format: each line "<node>,<boolean expression>".
    Boolean ops: ! for NOT, & for AND, | for OR.
    """
    with open(filename, "r") as f:
        data = [line.strip().split(separator) for line in f if line.strip()]

    nodes = [entry[0].strip() for entry in data]
    name_to_idx = {n: i for i, n in enumerate(nodes)}

    def convert_expression(expr: str) -> str:
        expr_py = expr.replace("!", " not ").replace("&", " and ").replace("|", " or ")
        def repl(m):
            tok = m.group(0)
            return f"x[{name_to_idx[tok]}]" if tok in name_to_idx else tok
        return re.sub(r"[A-Za-z0-9_\.]+", repl, expr_py)

    update_funcs = []
    for _, rule in data:
        expr_code = convert_expression(rule)
        func = eval(f"lambda x: {expr_code}", {"__builtins__": {}}, {})
        update_funcs.append(func)

    influence_map = [[] for _ in nodes]
    for tgt, (_, rule) in enumerate(data):
        deps = [name_to_idx[m.group(0)] for m in re.finditer(r"[A-Za-z0-9_\.]+", rule) if m.group(0) in name_to_idx]
        for src in deps:
            influence_map[src].append(tgt)

    return BooleanNetwork(nodes, name_to_idx, update_funcs, influence_map)


def async_update_step(state, update_functions):
    """
    Perform one asynchronous update: pick a random node whose next state differs.
    Returns index of updated node, or None if no changes needed.
    """
    to_update = [i for i, val in enumerate(state) if update_functions[i](state) != val]
    if not to_update:
        return None
    chosen = random.choice(to_update)
    state[chosen] = not state[chosen]
    return chosen


def run_to_stability(state, update_funcs, max_steps):
    """
    Run async updates until stable or step limit reached.
    Returns True if stable, False if max_steps exhausted.
    """
    for _ in range(max_steps):
        if async_update_step(state, update_funcs) is None:
            return True
    return False


def run_simulation(network, start_state, output_indices, max_steps=500):
    """
    Run a single simulation from start_state to stability.
    Returns tuple: (list[bool] of output node states, stabilized_flag).
    """
    state = start_state.copy()
    stabilized = run_to_stability(state, network.update_funcs, max_steps)
    outputs = [state[i] for i in output_indices]
    return outputs, stabilized


def bool_list_to_int(bool_list):
    """
    Encode a list of booleans as an integer bitmask.
    """
    return sum(1 << i for i, v in enumerate(bool_list) if v)


def int_to_active_nodes(number, output_indices, network):
    """
    Decode an integer bitmask to active node names.
    """
    active = []
    for bitpos, idx in enumerate(output_indices):
        if number & (1 << bitpos):
            active.append(network.node_names[idx])
    return active


def calculate_probabilities(results, output_indices, network):
    """
    Given list of integer-encoded outcomes, compute frequency probabilities.
    Returns dict mapping tuple(active_node_names) to probability.
    """
    counts = {}
    total = len(results)
    for r in results:
        counts[r] = counts.get(r, 0) + 1
    probs = {}
    for val, cnt in counts.items():
        names = tuple(int_to_active_nodes(val, output_indices, network))
        probs[names] = cnt / total
    return probs


def create_initial_state(network, activation_probs):
    """
    Generate a random initial Boolean state based on activation probabilities.
    activation_probs: dict[node_name, prob]
    """
    state = [False] * len(network.update_funcs)
    for name, p in activation_probs.items():
        idx = network.node_index.get(name)
        if idx is not None:
            state[idx] = random.random() < p
    return state


def create_mutant_network(bn, mutations):
    """
    Create a copy of `bn` with specified nodes fixed to constant values.
    mutations: list of (node_name: str, value: bool)
    """
    mut_dict = dict(mutations)
    # Copy network structure
    new_names = bn.node_names[:]
    new_index = bn.node_index.copy()
    new_funcs = []
    for name, fn in zip(bn.node_names, bn.update_funcs):
        if name in mut_dict:
            val = mut_dict[name]
            new_funcs.append(lambda x, v=val: v)
        else:
            new_funcs.append(fn)
    new_influence = bn.influence_map[:]
    return BooleanNetwork(new_names, new_index, new_funcs, new_influence)


def track_convergence(network, output_nodes, num_runs, max_steps,
                      start_probs, sample_interval=2):
    """
    Run multiple simulations, tracking outcome probabilities and stabilization stats.

    Returns:
      - history: list of (run_count, {outcome_names: probability})
        e.g, [(1, {(): 1.0}), 
              (2, {(): 0.5, ('CellCycleArrest', 'Invasion', 'EMT'): 0.5}), 
              (3, {(): 0.3333333333333333, ('CellCycleArrest', 'Invasion', 'EMT'): 0.3333333333333333, ('Apoptosis', 'CellCycleArrest'): 0.3333333333333333}), 
              (4, {(): 0.25, ('CellCycleArrest', 'Invasion', 'EMT'): 0.25, ('Apoptosis', 'CellCycleArrest'): 0.5}), 
              (5, {(): 0.4, ('CellCycleArrest', 'Invasion', 'EMT'): 0.2, ('Apoptosis', 'CellCycleArrest'): 0.4})
              ]
      - stats: dict with counts {'stable': int, 'unstable': int}
    """
    output_indices = [network.node_index[n] for n in output_nodes]
    counts = {}
    history = []
    stable_count = 0
    unstable_count = 0

    for run in range(1, num_runs + 1):
        state = create_initial_state(network, start_probs)
        out_bool, stabilized = run_simulation(network, state, output_indices, max_steps)
        if stabilized:
            stable_count += 1
        else:
            unstable_count += 1

        code = bool_list_to_int(out_bool)
        counts[code] = counts.get(code, 0) + 1

        if run % sample_interval == 0 or run == num_runs:
            probs = {tuple(int_to_active_nodes(k, output_indices, network)): v/run for k, v in counts.items()}
            history.append((run, probs))

    stats = {'stable': stable_count, 'unstable': unstable_count}
    return history, stats


def plot_convergence(prob_history, mutations=None):
    """
    Plot how outcome probabilities evolve over simulation runs.
    """
    title = "Probability Convergence"
    if mutations:
        mut_str = ", ".join(f"{g}={'ON' if v else 'OFF'}" for g, v in mutations)
        title = f"Mutant ({mut_str}) Probability Convergence"

    plt.figure(figsize=(10, 5))
    outcomes = set(o for _, ph in prob_history for o in ph.keys())
    for outcome in sorted(outcomes):
        runs = [r for r, _ in prob_history]
        probs = [ph.get(outcome, 0) for _, ph in prob_history]
        plt.plot(runs, probs, label=str(outcome))

    plt.xlabel("Number of Simulations")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
##########
## for {"ALPL":1, "HEY1":1, "SP7":1, "ADIPOQ":0, "FABP4":0, "CEBPA":0, "LPL":0}
##########

def pattern_to_int(pattern_dict, output_nodes, name_to_idx):
    """
    Given a dict {node_name:0/1} and a list of output_nodes in order,
    return the integer bitmask that corresponds to exactly that ON=1, OFF=0 pattern.
    """
    bits = []
    for node in output_nodes:
        val = pattern_dict.get(node, None)
        if val is None:
            raise KeyError(f"Pattern must specify a value for output node '{node}'")
        bits.append(bool(val))
    # use your existing bool_list_to_int
    return bool_list_to_int(bits)

def estimate_pattern_probability(network,
                                 output_nodes,
                                 start_probs,
                                 pattern_dict,
                                 num_runs=1000,
                                 max_steps=500):
    """
    Run `num_runs` simulations and return the fraction that end
    with exactly the ON/OFF pattern given by pattern_dict
    (keys = node names in output_nodes, values = 1 or 0).
    """
    # map output node names to indices
    output_indices = [network.node_index[n] for n in output_nodes]
    # precompute the integer code we want
    desired_code = pattern_to_int(pattern_dict, output_nodes, network.node_index)

    match_count = 0
    for _ in range(num_runs):
        init = create_initial_state(network, start_probs)
        out_bool, _ = run_simulation(network, init, output_indices, max_steps)
        code = bool_list_to_int(out_bool)
        if code == desired_code:
            match_count += 1

    return match_count / num_runs

