# nmcs_module.py
'''
import copy
import random
import time

# ------------------ Helper Functions ------------------ #

def is_terminal(state_set, depth):
    """Check if the mutation set has reached the desired size."""
    return len(state_set) >= depth

def legal_moves_fn(state_set, all_moves):
    """Return list of mutations not yet applied."""
    applied_genes = {gene for gene, val in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]

def random_playout(state_set, all_moves, depth, ec):
    """Complete the mutation sequence randomly up to `depth` and score it."""
    remaining = legal_moves_fn(state_set, all_moves)
    tail = random.sample(remaining, k=depth - len(state_set))
    full_random_set = list(state_set) + tail
    score = ec.evaluate(full_random_set)
    return score, full_random_set

# ------------------ NMCS Core ------------------ #

def nmcs(state, level, depth, best_moves, ec, timeout_sec=None):
    deadline = time.time() + timeout_sec if timeout_sec else None
    best = {'score': -float('inf'), 'state': []}
    return _nmcs(state, level, depth, best_moves, ec, {}, deadline, best)

def _nmcs(state, level, depth, best_moves, ec, cache, deadline, best):
    if deadline and time.time() > deadline:
        return best['score'], best['state']

    key = frozenset(state)
    if level == 0 or is_terminal(state, depth):
        if key in cache:
            score, s = cache[key]
        else:
            score, s = random_playout(state, best_moves, depth, ec)
            cache[key] = (score, s)
        if score > best['score']:
            best.update(score=score, state=s)
        return score, s

    moves = legal_moves_fn(state, best_moves)
    best_score, best_set = -float('inf'), []

    for m in moves:
        if deadline and time.time() > deadline:
            return best['score'], best['state']

        s1 = state + [m]
        key = frozenset(s1)

        if key in cache:
            score, s = cache[key]
        else:
            score, s = _nmcs(copy.deepcopy(s1), level-1, depth, moves, ec, cache, deadline, best)
            cache[key] = (score, s)

        if score > best_score:
            best_score, best_set = score, s
            if score > best['score']:
                best.update(score=score, state=s)

    return best_score, state + [best_set[len(state)]] if len(state) < len(best_set) else state
'''

# nmcs_module.py

import copy
import random
import time

# ------------------ Helper Functions ------------------ #

def normalize_sorted_list(state_list):
    """Return a NEW sorted list of (gene, bool) pairs."""
    return sorted(state_list, key=lambda x: (x[0], x[1]))

def normalize_key(state_list):
    """Hashable, order-independent cache key."""
    return tuple(normalize_sorted_list(state_list))

def is_terminal(state_set, depth):
    """Check if the mutation set has reached the desired size."""
    return len(state_set) >= depth

def legal_moves_fn(state_set, all_moves):
    """Return list of mutations not yet applied. Preserves order of all_moves."""
    applied_genes = {gene for gene, _ in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]

def random_playout(state_set, all_moves, depth, ec):
    """Complete the mutation set randomly up to `depth` and score it."""
    remaining = legal_moves_fn(state_set, all_moves)
    k = depth - len(state_set)
    k = max(0, min(k, len(remaining)))
    tail = random.sample(remaining, k=k)
    full_set = normalize_sorted_list(list(state_set) + tail)
    score = ec.evaluate(full_set)
    return score, full_set

# ------------------ NMCS Core ------------------ #
'''
def nmcs(state, level, depth, best_moves, ec, timeout_sec=None):
    deadline = time.time() + timeout_sec if timeout_sec else None
    best = {'score': -float('inf'), 'state': []}
    score, s = _nmcs(state, level, depth, best_moves, ec, {}, deadline, best)
    # Always return a sorted list for stable comparison
    return score, normalize_sorted_list(s)

def _nmcs(state, level, depth, best_moves, ec, cache, deadline, best):
    if deadline and time.time() > deadline:
        # Return whatever best we have so far
        return best['score'], best['state']

    # Normalize key for cache (order-independent)
    key = normalize_key(state)

    # Base case or terminal: complete randomly and update best
    if level == 0 or is_terminal(state, depth):
        if key in cache:
            score, s = cache[key]
        else:
            score, s = random_playout(state, best_moves, depth, ec)
            cache[key] = (score, s)
        if score > best['score']:
            best.update(score=score, state=s)
        return score, s

    # Explore children
    moves = legal_moves_fn(state, best_moves)
    best_score_here, best_set_here = -float('inf'), []

    for m in moves:
        if deadline and time.time() > deadline:
            return best['score'], best['state']

        # Keep state sorted during search for stable comparisons
        s1 = normalize_sorted_list(list(state) + [m])
        k1 = normalize_key(s1)

        if k1 in cache:
            score, s = cache[k1]
        else:
            score, s = _nmcs(s1, level-1, depth, best_moves, ec, cache, deadline, best)
            cache[k1] = (score, s)

        if score > best_score_here:
            best_score_here, best_set_here = score, s
            if score > best['score']:
                best.update(score=score, state=s)

    # IMPORTANT: return the full best set, not just the next move
    return best_score_here, best_set_here
'''
# ------------------ NMCS Core (per-level cache + expand-by-best loop) ------------------ #

def nmcs(state, level, depth, best_moves, ec, timeout_sec=None):
    deadline = time.time() + timeout_sec if timeout_sec else None
    best = {'score': -float('inf'), 'state': []}

    caches = [dict() for _ in range(level + 1)]

    score, s = _nmcs(state, level, depth, best_moves, ec, caches, deadline, best)
    return score, normalize_sorted_list(s)


def _nmcs(state, level, depth, best_moves, ec, caches, deadline, best):
    if deadline and time.time() > deadline:
        return best['score'], best['state']

    # order-free
    key = normalize_key(state)

    if key in caches[level]:
        return caches[level][key]

    if level == 0 or is_terminal(state, depth):
        if key in caches[0]:
            score, s = caches[0][key]
        else:
            score, s = random_playout(state, best_moves, depth, ec)
            caches[0][key] = (score, s)
        if score > best['score']:
            best.update(score=score, state=s)
        return score, s

    s_cur = normalize_sorted_list(list(state))
    best_score_here, best_set_here = -float('inf'), []

    while not is_terminal(s_cur, depth): # main loop: expand-by-best
        if deadline and time.time() > deadline:
            return best['score'], best['state']

        moves = legal_moves_fn(s_cur, best_moves)
        local_best_score, local_best_set = -float('inf'), []

        for m in moves: # explore children
            if deadline and time.time() > deadline:
                return best['score'], best['state']

            s1 = normalize_sorted_list(s_cur + [m])
            k1 = normalize_key(s1)

            # children use caches[level-1]
            if k1 in caches[level - 1]:
                score, s = caches[level - 1][k1]
            else:
                score, s = _nmcs(s1, level - 1, depth, best_moves, ec, caches, deadline, best)
                caches[level - 1][k1] = (score, s)

            if score > local_best_score:
                local_best_score, local_best_set = score, s

            if score > best['score']:
                best.update(score=score, state=s)

        if local_best_score > best_score_here:
            best_score_here, best_set_here = local_best_score, local_best_set

        # S <- S ∪ { next(bestSet \ S) }
        next_elem = None
        for x in best_set_here:
            if x not in s_cur:
                next_elem = x
                break
        s_cur = normalize_sorted_list(s_cur + [next_elem])
    return best_score_here, s_cur

