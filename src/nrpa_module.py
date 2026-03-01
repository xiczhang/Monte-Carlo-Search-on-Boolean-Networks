import time
import copy
import math
import random
import numpy as np

def is_terminal(state_set, depth):
    return len(state_set) >= depth

def legal_moves_fn(state_set, all_moves):
    applied_genes = {gene for gene, _ in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]

def code(state, m):
    if not isinstance(state, list):
        raise TypeError(f"Expected state to be list, got {type(state).__name__!r}")
    if not isinstance(m, tuple):
        raise TypeError(f"Expected m to be tuple, got {type(m).__name__!r}")
    return hash((tuple(sorted(state)), m))

def randomMove(state_set, all_moves, policy):
    moves = legal_moves_fn(state_set, all_moves)
    weights = [math.exp(policy.get(code(state_set, m), 0.0)) for m in moves]
    total = sum(weights)
    stop = random.random() * total

    cumulative = 0.0
    for m, w in zip(moves, weights):
        cumulative += w
        if cumulative >= stop:
            return m
    return moves[-1]  # fallback

def play(state, move):
    return state + [move]

def playout(state, policy, depth, ec, all_moves):
    state = list(state)
    while not is_terminal(state, depth):
        m = randomMove(state, all_moves, policy)
        state = play(state, m)
    return ec.evaluate(sorted(state)), sorted(state)

def adapt(policy, state_set, all_moves):
    new_policy = copy.deepcopy(policy)
    state = []

    for move in state_set:
        moves = legal_moves_fn(state, all_moves)
        Z = sum(math.exp(policy.get(code(state, m), 0.0)) for m in moves)

        for m in moves:
            c = code(state, m)
            prob = math.exp(policy.get(c, 0.0)) / Z
            new_policy[c] = new_policy.get(c, 0.0) - prob

        c_best = code(state, move)
        new_policy[c_best] = new_policy.get(c_best, 0.0) + 1.0
        state = play(state, move)

    return new_policy

def nrpa(level, policy, depth, ec, all_moves, timeout_sec=None):
    deadline = time.time() + timeout_sec if timeout_sec else None
    best_global = {'score': -float('inf'), 'state': []}
    return _nrpa(level, policy, depth, ec, all_moves, deadline, best_global)

def _nrpa(level, policy, depth, ec, all_moves, deadline, best_global):
    if deadline and time.time() > deadline:
        return best_global['score'], best_global['state']

    if level == 0:
        score, state = playout([], policy, depth, ec, all_moves)
        if score > best_global['score']:
            best_global.update(score=score, state=state)
        return score, state

    best = -float('inf')
    best_set = []

    for _ in range(9999):
        if deadline and time.time() > deadline:
            break

        pol = copy.deepcopy(policy)
        sc, s = _nrpa(level - 1, pol, depth, ec, all_moves, deadline, best_global)

        if sc > best:
            best = sc
            best_set = s
            if sc > best_global['score']:
                best_global.update(score=sc, state=s)

        policy = adapt(policy, best_set, all_moves)

    return best, best_set
