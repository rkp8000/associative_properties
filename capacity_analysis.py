"""
functions for determining capacity of random network
"""
from __future__ import division, print_function
import numpy as np
from scipy import optimize
from scipy import stats


def _log_sum(log_xs):
    """Efficiently calculates the logarithm of a sum from the logarithm of its
    terms.

    logsum employs an efficient algorithm for finding the logarithm of a sum
    of numbers when provided with the logarithm of the terms in the sum; this
    is useful when the logarithms are too large or small
    and would cause numerical errors if exponentiated

    :param log_xs: logarithms of terms to sum (1d array)
    :return logarithm of the sum

    example:
        >>> log_xs = np.array([-1000,-1001,-1002])
        >>> np.log(np.sum(np.exp(log_xs)))
        -inf
        >>> _log_sum(log_xs)
        -999.59239403555557
        >>> log_xs = np.array([1000,1001,1002])
        >>> np.log(np.sum(np.exp(log_xs)))
        inf
        >>> _log_sum(log_xs)
        1002.4076059644444
    """
    # get largest element

    log_x_max = np.max(log_xs)

    # normalize log_x by subtracting log_x_max

    log_x_normed = log_xs - log_x_max

    # calculate sum of logarithms

    return log_x_max + np.log(np.sum(np.exp(log_x_normed)))


def no_interference_first_items(vs):
    """
    For each of several matrices, calculate whether they will interfere with one another or not
    :param vs: stack of 2D arrays corresponding to connections for first 2L items
    :return: 1 or 0 for each v; 1 indicates no interference among first items, 0 otherwise
    """

    # calculate maintained sets

    fs = -1 * np.ones((vs.shape[0],), dtype=int)

    l = int(vs.shape[1] / 2)

    for v_ctr, v in enumerate(vs):

        # get intersections, sizes, and maintained sets

        isctns = np.array([v[2*i, :] * v[2*i+1, :] for i in range(int(len(v)/2))])
        isctn_sizes = isctns.sum(1)
        maintained = (isctns.sum(0) > 0).astype(int)

        # set to 0 if any empty intersections

        if np.any(isctn_sizes == 0):

            fs[v_ctr] = 0
            continue

        f = 1

        # loop over all intersections

        for i, isctn_size in enumerate(isctn_sizes):

            # loop over all items not in this conjunction

            for j in [jj for jj in range(l) if jj not in [2*i, 2*i + 1]]:

                # set to 0 if this item interferes with either item in current conjunction

                if np.sum(v[j, :] * v[2*i, :] * maintained) >= isctn_size:

                    f = 0
                    break

                if np.sum(v[j, :] * v[2*i + 1, :] * maintained) >= isctn_size:

                    f = 0
                    break

            # stop counting if an interference has been found

            if f == 0:

                break

        fs[v_ctr] = f

    return fs


def log_neg_log_probability_no_interference_lower_bound_random_item(vs, q):
    """
    Calculate a lower bound on the probability that there will be no interference between
    the random connections between a single item and the association/memory reservoir,
    given the connections of a set of item associations to be remembered.

    :param vs: list of vs (connections between first 2l item units and memory reservoir)
    :param q: connection probability
    :return: log of the negative log of lower bound on probability of no interference for each v
    """

    log_neg_log_hs = np.nan * np.zeros((vs.shape[0],))

    for v_ctr, v in enumerate(vs):

        # get intersections, sizes, and maintained sets

        isctns = np.array([v[2 * i, :] * v[2 * i + 1, :] for i in range(int(len(v) / 2))])
        isctn_sizes = isctns.sum(1)
        maintained = (isctns.sum(0) > 0).astype(int)

        # store the log of the negative log of each cumulative distribution

        temps = []

        for i, isctn_size in enumerate(isctn_sizes):

            log_sf = stats.binom.logsf(isctn_size - 1, np.sum(v[i, :] * maintained), q)

            # if sf is very small then sf = -log(cdf) approximately

            if log_sf < -9*np.log(10):

                temps.append(log_sf)

            else:

                log_cdf = stats.binom.logcdf(isctn_size - 1, np.sum(v[i, :] * maintained), q)
                temps.append(np.log(-log_cdf))

        log_neg_log_hs[v_ctr] = _log_sum(np.array(temps))

    return log_neg_log_hs


def recall_error_upper_bound(log_m, l, fs, log_neg_log_hs):
    """
    calculate an upper bound on the recall error for an idealized version of the short-term
    associative memory network with random connections; this function assumes that f and h have been
    precomputed from a random selection of {v_1, ..., v_2l}; as the size of f, h, increases the approximation
    becomes exact.

    :param log_m: log of number of item units
    :param l: number of pair-wise associations required to recall
    :param fs: indicator values indicating whether {v_1, ..., v_2L} interferes with recall
    :param log_neg_log_hs: log of negative log probability that a single random v_j interferes with recall
        (see notebook for why this is a useful mathematical quantity)
    :return approximate probability of incorrectly recalling l associations
    """

    # assume m = m - 2L if m/(2L) > 1e9

    if (log_m - np.log(2*l)) > 9*np.log(10):

        temp_0 = log_m

    else:

        temp_0 = np.log(np.exp(log_m) - (2*l))

    # calc temp quantity comparing m to

    temp_1 = temp_0 + log_neg_log_hs

    # calculate error using either exact formula or log approximation

    errs = np.nan * np.zeros(temp_1.shape)

    approx_mask = temp_1 < np.log(-np.log(1 - 1e-9))

    errs[~approx_mask] = 1 - np.exp(-np.exp(temp_1[~approx_mask]))
    errs[approx_mask] = np.exp(temp_1[approx_mask])

    # set error to 1 for deterministic errors

    errs[fs == 0] = 1

    return np.mean(errs)


def recall_error_upper_bound_vs_item_number(ms, n, q, l, n_samples_mc, vs=None):
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

    if vs is None:

        vs = (np.random.rand(n_samples_mc, 2*l, n) < q).astype(int)

    # calculate f, the indicator values for whether the first 2l item unit connections
    # interfere with recall (with an indicator of 0 if they do)

    fs = no_interference_first_items(vs)

    # calculate h, a lower bound probability that a random item's connections interfere

    log_neg_log_hs = log_neg_log_probability_no_interference_lower_bound_random_item(vs, q)

    # calculate lower bound on correct recall probabilities

    errs = [recall_error_upper_bound(np.log(m), l, fs, log_neg_log_hs) for m in ms]

    return np.array(errs)


def log_max_items_with_low_recall_error(n, q, l, err_max, n_samples_mc, m_tol, vs=None):
    """
    approximate the maximum number of items that can be randomly connected to a memory reservoir such
    that the probability of incorrect recall is less than a specified level

    :param n: number of association/memory units
    :param q: connection probability
    :param l: number of associations to recall
    :param err_max: max recall error (from 0 to 1)
    :param n_samples_mc: number of samples to use in monte carlo simulation
    :return: number of items (m)
    """

    # first take many samples of connections of the first 2l items to the memory units

    if vs is None:

        vs = (np.random.rand(n_samples_mc, 2 * l, n) < q).astype(int)

    # calculate f, the indicator values for whether the first 2l item unit connections
    # interfere with recall (with an indicator of 0 if they do)

    fs = no_interference_first_items(vs)

    # calculate h, a lower bound probability that a random item's connections interfere

    log_neg_log_hs = log_neg_log_probability_no_interference_lower_bound_random_item(vs, q)

    # determine bounds for the function solver

    ## keep doubling m until the recall probability is lower than p_min

    log_m_test = np.log(2*l + 1)

    if recall_error_upper_bound(log_m_test, l, fs, log_neg_log_hs) >= err_max:

        return 0

    while recall_error_upper_bound(log_m_test, l, fs, log_neg_log_hs) < err_max:

        log_m_test += 1

    # now that we have upper and lower bounds, solve for the best m using Brent's method

    def function_to_solve(log_m):

        return recall_error_upper_bound(log_m, l, fs, log_neg_log_hs) - err_max

    log_m_best = optimize.brentq(function_to_solve, log_m_test - 1, log_m_test, xtol=m_tol)

    return log_m_best
