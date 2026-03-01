
import math
import random
import time
# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def legal_moves_fn(state_set, all_moves):
    applied_genes = {gene for gene, _ in state_set}
    return [m for m in all_moves if m[0] not in applied_genes]

def code_fn(state, move):
    return hash((tuple(sorted(state)), move))

def softmax_probs(logits, tau):
    expv = [math.exp(l / tau) for l in logits]
    Z = sum(expv)
    return [v / Z for v in expv], Z


# ----------------------------------------------------------------------------
# 1) Level-0 playout with trace recording (Gibbs-style)
# ----------------------------------------------------------------------------


def gnrpa_playout_and_trace(policy, bias, tau, depth, ec, all_moves):
    state, sequence = [], []
    code_matrix, index_list, O_list, Z_list = [], [], [], []

    while len(state) < depth:
        applied_genes = {gene for gene, _ in state}
        moves = [m for m in all_moves if m[0] not in applied_genes]
        if not moves:
            break

        codes = [hash((tuple(sorted(state)), m)) for m in moves]
        ws = [policy.get(c, 0.0) for c in codes]
        bs = [bias.get(c, 0.0) for c in codes]

        o = [math.exp((w + b) / tau) for w, b in zip(ws, bs)]
        Z = sum(o)
        if Z == 0.0:
            break

        chosen_j = random.choices(range(len(o)), weights=o)[0]

        code_matrix.append(codes)
        index_list.append(chosen_j)
        O_list.append(o)
        Z_list.append(Z)

        chosen_move = moves[chosen_j]
        state.append(chosen_move)
        sequence.append(chosen_move)

    score = ec.evaluate(sequence)
    return score, sequence, code_matrix, index_list, O_list, Z_list


def gnrpa_adapt_inplace(policy, code_matrix, index_list, O_list, Z_list, tau):
    factor = tau
    for i, codes_i in enumerate(code_matrix):
        best_j = index_list[i]
        oi, zi = O_list[i], Z_list[i]
        for j, c in enumerate(codes_i):
            policy.setdefault(c, 0.0)
            pij = oi[j] / zi
            delta = pij - (1.0 if j == best_j else 0.0)
            policy[c] -= factor * delta
    return policy


def gnrpa(level, policy, bias, tau, depth, ec, all_moves, N=100, timeout_sec=None):
    deadline = time.time() + timeout_sec if timeout_sec else None
    return _gnrpa(level, policy, bias, tau, depth, ec, all_moves, N, deadline)


def _gnrpa(level, policy, bias, tau, depth, ec, all_moves, N, deadline):
    if level == 0:
        return gnrpa_playout_and_trace(policy, bias, tau, depth, ec, all_moves)

    best_score = float("-inf")
    best_seq = None
    best_traces = None

    for _ in range(N):
        if deadline and time.time() > deadline:
            break

        child_policy = policy.copy()
        sc, seq, cm, il, ol, zl = _gnrpa(
            level - 1, child_policy, bias, tau, depth, ec, all_moves, N, deadline
        )

        if sc > best_score:
            best_score = sc
            best_seq = seq
            best_traces = (cm, il, ol, zl)

        if best_traces is not None:
            gnrpa_adapt_inplace(policy, *best_traces, tau)

    if best_traces is not None:
        cm, il, ol, zl = best_traces
    else:
        cm, il, ol, zl = [], [], [], []

    return best_score, best_seq, cm, il, ol, zl
