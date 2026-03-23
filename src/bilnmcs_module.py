# bilnmcs_module.py
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
    # Guard (should be safe if depth ≤ total unique genes)
    k = max(0, min(k, len(remaining)))
    tail = random.sample(remaining, k=k)
    full_set = normalize_sorted_list(list(state_set) + tail)
    score = ec.evaluate(full_set)
    return score, full_set
# ------------------ LNMCS (Lazy NMCS) ------------------ #


def bilnmcs(state, level, depth, all_moves, ec_main, ec_fast, *,
            b=2, r=0.5, e=None, timeout_sec=None):
    """
    Bi-Lazy NMCS entrypoint with per-level caches and expand-by-best loop.
    """
    deadline = time.time() + timeout_sec if timeout_sec else None

    # per-depthh running stats for thresholds
    max_depth = depth
    tr    = [{'mean': 0.0, 'count': 0} for _ in range(max_depth + 1)]
    trmax = [float('-inf') for _ in range(max_depth + 1)]

    # Per-level caches (main ans fast )
    c_main = [dict() for _ in range(level + 1)]
    c_fast = [dict() for _ in range(level + 1)]

    best = {'score': -float('inf'), 'state': []}
    score, s = _bilnmcs(state, level, depth, all_moves, ec_main, ec_fast,
                        c_main=c_main, c_fast=c_fast, deadline=deadline, best=best,
                        tr=tr, trmax=trmax, b=b, r=r, e=e)
    return score, normalize_sorted_list(s)


def _bilnmcs(state, level, depth, all_moves, ec_main, ec_fast, *,
             c_main, c_fast, deadline, best, tr, trmax, b, r, e):
    # Timeout 
    if deadline and time.time() > deadline:
        return best['score'], best['state']

    state = normalize_sorted_list(list(state))
    key   = normalize_key(state)

    if key in c_main[level]:
        return c_main[level][key]

    #  playout, cached at level 0
    if level == 0 or is_terminal(state, depth):
        if key in c_main[0]:
            score, s = c_main[0][key]
        else:
            score, s = random_playout(state, all_moves, depth, ec_main)
            c_main[0][key] = (score, s)
        if score > best['score']:
            best.update(score=score, state=s)
        return score, s

    bestSc_level = -float('inf')
    bestSet_level = []
    S_cur = list(state)  # we’ll grow this with next(bestSet \ S)

    while not is_terminal(S_cur, depth):
        if deadline and time.time() > deadline:
            return best['score'], best['state']

        # (i) Legal moves (cap to e)
        moves = legal_moves_fn(S_cur, all_moves)
        if not moves:
            break
        if e is not None and len(moves) > e:
            moves = random.sample(moves, e)

        # (ii) ccheap evals for each move using ec_fast, cached in c_fast[0]
        d = min(len(S_cur), depth)
        candidates = []  # (mean_eval, move)
        for m in moves:
            if deadline and time.time() > deadline:
                return best['score'], best['state']

            S1  = normalize_sorted_list(S_cur + [m])
            k1  = normalize_key(S1)
            tot = 0.0
            for _ in range(max(1, b)):
                if k1 in c_fast[0]:
                    sc, ss = c_fast[0][k1]
                else:
                    sc, ss = random_playout(S1, all_moves, depth, ec_fast)
                    c_fast[0][k1] = (sc, ss)
                tot += sc
                if sc > best['score']:
                    best.update(score=sc, state=ss)
            mean_eval = tot / max(1, b)
            candidates.append((mean_eval, m))

            # per-depth running stats
            acc = tr[d]
            acc['mean'] = (acc['mean'] * acc['count'] + mean_eval) / (acc['count'] + 1)
            acc['count'] += 1
            if mean_eval > trmax[d]:
                trmax[d] = mean_eval

        # Depth-specific threshold
        mu_d  = tr[d]['mean']
        b_d   = trmax[d]
        theta = mu_d + r * (b_d - mu_d)

        # Lazy recursion (prune -> level 0; else -> level-1), track best child-set
        local_best_sc, local_best_set = -float('inf'), []
        for mean_eval, m in candidates:
            if deadline and time.time() > deadline:
                return best['score'], best['state']

            S1 = normalize_sorted_list(S_cur + [m])
            k1 = normalize_key(S1)
            next_level = 0 if mean_eval < theta else (level - 1)

            if next_level == 0:
                if k1 in c_main[0]:
                    sc, s = c_main[0][k1]
                else:
                    sc, s = random_playout(S1, all_moves, depth, ec_main)
                    c_main[0][k1] = (sc, s)
            else:
                if k1 in c_main[level - 1]:
                    sc, s = c_main[level - 1][k1]
                else:
                    sc, s = _bilnmcs(S1, level - 1, depth, all_moves, ec_main, ec_fast,
                                     c_main=c_main, c_fast=c_fast, deadline=deadline, best=best,
                                     tr=tr, trmax=trmax, b=b, r=r, e=e)
                    c_main[level - 1][k1] = (sc, s)

            if sc > local_best_sc:
                local_best_sc, local_best_set = sc, s
            if sc > bestSc_level:
                bestSc_level, bestSet_level = sc, s
            if sc > best['score']:
                best.update(score=sc, state=s)

        # S_cur <- S_cur ∪ { next(bestSet \ S_cur) }
        x_star = next((x for x in local_best_set if x not in S_cur), None)
        if x_star is None:
            break  
        S_cur = normalize_sorted_list(S_cur + [x_star])
        
    return bestSc_level, S_cur
