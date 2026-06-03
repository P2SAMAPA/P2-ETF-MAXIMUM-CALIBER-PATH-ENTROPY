import numpy as np
from scipy.stats import entropy

def discretise_returns(returns, n_bins=3):
    """Discretise returns into n_bins equal‑width bins."""
    if len(returns) < 2:
        return np.array([])
    try:
        bins = np.percentile(returns, np.linspace(0, 100, n_bins+1)[1:-1])
    except:
        bins = []
    if len(bins) == 0:
        min_r, max_r = returns.min(), returns.max()
        bins = np.linspace(min_r, max_r, n_bins+1)[1:-1]
    states = np.digitize(returns, bins)
    return states

def discretise_macro(macro_series, n_bins=2):
    """Discretise macro variable (e.g., VIX) into n_bins quantiles."""
    if len(macro_series) < n_bins:
        return np.zeros_like(macro_series, dtype=int)
    bins = np.percentile(macro_series, np.linspace(0, 100, n_bins+1)[1:-1])
    return np.digitize(macro_series, bins)

def entropy_rate_markov_conditional(states, macro_states, order=1):
    """
    Compute entropy rate of a Markov chain conditioned on macro state.
    H = sum_{macro} p(macro) * H( next | past, macro )
    """
    if len(states) < order+1:
        return 0.0
    # Compute joint with macro
    n_macro = np.max(macro_states) + 1
    total_H = 0.0
    total_weight = 0.0
    for m in range(n_macro):
        idx = (macro_states == m)
        if np.sum(idx) < order+1:
            continue
        states_sub = states[idx]
        # Compute transition counts for this macro
        transition_counts = {}
        hist_counts = {}
        for i in range(len(states_sub) - order):
            hist = tuple(states_sub[i:i+order])
            nxt = states_sub[i+order]
            key = (hist, nxt)
            transition_counts[key] = transition_counts.get(key, 0) + 1
            hist_counts[hist] = hist_counts.get(hist, 0) + 1
        if len(hist_counts) == 0:
            continue
        H_m = 0.0
        total_trans = sum(transition_counts.values())
        for (hist, nxt), cnt in transition_counts.items():
            p_hist = hist_counts[hist] / total_trans
            p_nxt_given_hist = cnt / hist_counts[hist]
            H_m += p_hist * (-p_nxt_given_hist * np.log2(p_nxt_given_hist))
        # Weight by proportion of macro
        p_m = np.sum(idx) / len(states)
        total_H += p_m * H_m
        total_weight += p_m
    if total_weight == 0:
        return 0.0
    return total_H / total_weight

def path_entropy_score(returns, macro_series=None, n_bins=3, entropy_type='rate', markov_order=1, use_macro=False, macro_bins=2):
    """
    Score = entropy rate (or block entropy) of the return path.
    If use_macro and macro_series provided, condition on macro states.
    """
    states = discretise_returns(returns, n_bins)
    if len(states) < markov_order + 1:
        return 0.0
    if use_macro and macro_series is not None:
        macro_states = discretise_macro(macro_series, macro_bins)
        if entropy_type == 'rate':
            H = entropy_rate_markov_conditional(states, macro_states, markov_order)
        else:
            # For block entropy, we could also condition, but simpler: return joint entropy
            H = entropy_rate_markov_conditional(states, macro_states, markov_order)  # placeholder
    else:
        if entropy_type == 'rate':
            H = entropy_rate_markov(states, markov_order)
        else:
            H = block_entropy(states, markov_order)
    return float(H)

def entropy_rate_markov(states, order=1):
    """Unconditional entropy rate (for fallback)."""
    n_states = np.max(states) + 1
    transition_counts = {}
    hist_counts = {}
    for i in range(len(states) - order):
        hist = tuple(states[i:i+order])
        nxt = states[i+order]
        key = (hist, nxt)
        transition_counts[key] = transition_counts.get(key, 0) + 1
        hist_counts[hist] = hist_counts.get(hist, 0) + 1
    if len(hist_counts) == 0:
        return 0.0
    H = 0.0
    total_trans = sum(transition_counts.values())
    for (hist, nxt), cnt in transition_counts.items():
        p_hist = hist_counts[hist] / total_trans
        p_nxt_given_hist = cnt / hist_counts[hist]
        H += p_hist * (-p_nxt_given_hist * np.log2(p_nxt_given_hist))
    return H

def block_entropy(states, order=1):
    """Block entropy (joint entropy) for fallback."""
    if len(states) < order:
        return 0.0
    block_counts = {}
    for i in range(len(states) - order + 1):
        block = tuple(states[i:i+order])
        block_counts[block] = block_counts.get(block, 0) + 1
    total = len(states) - order + 1
    probs = np.array(list(block_counts.values())) / total
    return entropy(probs, base=2)
