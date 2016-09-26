"""
functions for determining capacity of random network
"""
from __future__ import division, print_function
import numpy as np
from scipy import stats


def recall_probability_lower_bound_vs_item_number(ms, n, q, l, n_samples_mc):
    """
    calculate using Monte Carlo simulation the probability of correctly recalling l associations
    between m possible items connected to n association/memory units
    :param ms: number of item units (1D array)
    :param n: number of association/memory units
    :param q: connection probability
    :param l: number of associations to recall
    :param n_samples_mc: number of samples in monte carlo simulation
    :return: 1D array of probabilities (one for each m)
    """

    # first take many samples of connections of the first 2l items to the memory units

    vs = (np.random.rand(n_samples_mc, 2*l, n) < q).astype(int)

    # calculate f, the indicator values for whether the first 2l item unit connections
    # interfere with recall (with an indicator of 0 if they do)

    fs = no_interference_first_items(vs)

    # calculate h, a lower bound probability that a random item's connections interfere

    log_hs = log_probability_no_interference_lower_bound_random_item(vs, q)

    # calculate lower bound on correct recall probabilities

    ps = [recall_probability_lower_bound(m, l, fs, log_hs) for m in ms]

    return np.array(ps)


def max_items_with_min_recall_probability(n, q, l, p_min, n_samples_mc):
    """
    approximate the maximum number of items that can be randomly connected to a memory reservoir such
    that the probability of correctly recalling l pair-wise associations is greater than a
    minimum value
    :param n: number of association/memory units
    :param q: connection probability
    :param l: number of associations to recall
    :param p_min: minimum recall probability
    :param n_samples_mc: number of samples to use in monte carlo simulation
    :return: number of items (m)
    """

    # first take many samples of connections of the first 2l items to the memory units

    vs = (np.random.rand(n_samples_mc, 2 * l, n) < q).astype(int)

    # calculate f, the indicator values for whether the first 2l item unit connections
    # interfere with recall (with an indicator of 0 if they do)

    fs = no_interference_first_items(vs)

    # calculate h, a lower bound probability that a random item's connections interfere

    log_hs = log_probability_no_interference_lower_bound_random_item(vs, q)

    def function_to_solve(m):

        return recall_probability_lower_bound(m ,l, fs, log_hs) - p_min

    # solve the function

    # keep doubling m until function_to_solve spits out a value greater than 0

    m_test = 2*l

    m_lb = None
    m_ub = None

    # now that we have upper and lower bounds, solve for the best m

    m_best = None

    return m_best


def recall_probability_lower_bound(m, l, fs, log_hs):
    """
    calculate a lower bound on the correct recall probability for an idealized version of the short-term
    associative memory network with random connections; this function assumes that f and h have been
    precomputed from a random selection of {v_1, ..., v_2l}; as the size of f, h, increases the approximation
    becomes exact.

    :param m: number of item units
    :param l: number of pair-wise associations required to recall
    :param fs: indicator values indicating whether {v_1, ..., v_2L} interferes with recall
    :param log_hs: log probability that a single random v_j interferes with recall
    :return approximate probability that l associations can be recalled precisely
    """

    return np.exp((m - (2*l)) * log_hs) * fs


def no_interference_first_items(vs):
    """
    For each of several matrices, calculate whether they will interfere with one another or not
    :param vs:
    :return:
    """

    # calculate maintained sets

    fs = -1 * np.ones((vs.shape[0],), dtype=int)

    l = int(vs.shape[1] / 2)

    for v_ctr, v in enumerate(vs):

        # get intersections, sizes, and maintained sets

        isctns = np.array([v[2*i, :] * v[2*i+1, :] for i in range(int(len(v)/2))])
        isctn_sizes = isctns.sum(1)
        maintained = (isctns.sum(0) > 0).astype(int)

        if np.any(isctn_sizes == 0):

            fs[v_ctr] = 0
            continue

        f = 1

        for i, isctn_size in enumerate(isctn_sizes):

            for j in [jj for jj in range(l) if jj not in [2*i, 2*i + 1]]:

                if np.sum(v[j, :] * v[2*i, :] * maintained) >= isctn_size:

                    f = 0

                    break

                if np.sum(v[j, :] * v[2*i + 1, :] * maintained) >= isctn_size:

                    f = 0

                    break

            if f == 0:

                break

        fs[v_ctr] = f

    return fs


def log_probability_no_interference_lower_bound_random_item(vs, q):
    """
    Calculate a lower bound on the probability that there will be no interference between
    the random connections between a single item and the association/memory reservoir,
    given the connections of a set of item associations to be remembered.
    between a
    :param vs: list of vs (connections between first 2l item units and memory reservoir)
    :param q: connection probability
    :return: probability of no interference for each v
    """

    log_hs = np.nan * np.zeros((vs.shape[0],))

    for v_ctr, v in enumerate(vs):

        log_h = 0

        # get intersections, sizes, and maintained sets

        isctns = np.array([v[2 * i, :] * v[2 * i + 1, :] for i in range(int(len(v) / 2))])
        isctn_sizes = isctns.sum(1)
        maintained = (isctns.sum(0) > 0).astype(int)

        for i, isctn_size in enumerate(isctn_sizes):

            log_h += stats.binom.logcdf(isctn_size - 1, np.sum(v[i, :] * maintained), q)

        log_hs[v_ctr] = log_h

    return log_hs
