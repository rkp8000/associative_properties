"""
functions for determining capacity of random network
"""
from __future__ import division, print_function
import numpy as np
from scipy import optimize
from scipy import stats

import auxiliary as aux


def max_items_low_error(max_log_error, n_mc, n, l, q, r=None, log_epsilon=-9*np.log(10)):
    """
    Calculate the log of the maximum number of items that a set of associative
    memory units can support with the expected recall error lying below
    a specified quantity.
    """

    # calculate the main components of the terms going into the
    # monte-carlo sampling sum

    fs, log_neg_log_hs = calc_fs_and_log_neg_log_hs(n_mc, n, l, q, r, log_epsilon)

    # make function that return diff between true and acceptable error

    def func_to_solve(log_m):

        log_mc_sum = calc_log_mc_sum(fs, log_neg_log_hs, log_m, l, log_epsilon) - np.log(n_mc)

        return log_mc_sum - max_log_error

    # determine initial optimization bracket

    log_m_ub = np.log(2*l)

    # if 2L item units gives too large an error, return -np.inf

    if func_to_solve(log_m_ub) >= 0: return -np.inf
    else: log_m_ub += 1

    while func_to_solve(log_m_ub) < 0: log_m_ub += 1

    # find the root of the equation

    return optimize.brentq(func_to_solve, log_m_ub - 1, log_m_ub)


def log_upper_error_bound(log_ms, n_mc, n, l, q, r=None, log_epsilon=-9*np.log(10)):
    """
    Calculate the log upper bound of the expected recall error for a network
    with n units and several different numbers of item units connected.
    """

    fs, log_neg_log_hs = calc_fs_and_log_neg_log_hs(n_mc, n, l, q, r, log_epsilon)

    log_errors = [
        calc_log_mc_sum(fs, log_neg_log_hs, log_m, l, log_epsilon) - np.log(n_mc)
        for log_m in log_ms
    ]

    return log_errors


def calc_log_mc_sum(fs, log_neg_log_hs, log_m, l, log_epsilon):
    """
    Calculate the log of the sum of the terms in the Monte-Carlo approximation
    given some temporary quantities.
    """

    # approximate log(M - 2L) if necessary

    if np.log(2*l) - log_m < log_epsilon: log_m_minus_2l = log_m
    else: log_m_minus_2l = np.log(np.exp(log_m) - 2*l)

    # make mask for approximating log(M - 2L) + log(-log(h))

    approx_mask = log_m_minus_2l + log_neg_log_hs < log_epsilon

    # get log of all terms in sum

    log_sum_terms = np.nan * np.zeros((len(fs),))

    # fill in terms requiring approximation

    log_sum_terms[approx_mask] = log_m_minus_2l + log_neg_log_hs[approx_mask]

    # fill in terms not requiring approximation

    h_to_m_minus_2ls = np.exp(-np.exp(log_neg_log_hs[~approx_mask])) ** np.exp(log_m_minus_2l)
    log_sum_terms[~approx_mask] = np.log(1 - h_to_m_minus_2ls)

    # set log error for terms with f = 0 (interference among first items) to 0

    log_sum_terms[fs == 0] = 0

    return aux.log_sum(log_sum_terms)


def calc_fs_and_log_neg_log_hs(n_mc, n, l, q, r, log_epsilon):
    """
    Calculate the f and h components of each term in the Monte-Carlo sum.
    """

    # sample first 2L units' cxns and calculate the pairwise intersections
    # and their sizes

    if r is None:  # symmetric case

        vs = sample_vs(n_mc=n_mc, n=n, l=l, q=q)
        xs_all, rs_all = zip(*[calc_xs_and_rs(v) for v in vs])

    else:  # asymmetric case

        vs, us = sample_vs_and_us(n_mc=n_mc, n=n, l=l, q=q, r=r)
        xs_all, rs_all = zip(*[calc_xs_and_rs(v, u) for v, u in zip(vs, us)])

    x_sizes_all = [xs.sum(axis=1) for xs in xs_all]

    # calc f and log(-log(h)) for each v

    fs = [calc_f(v, xs, rs) for v, xs, rs in zip(vs, xs_all, rs_all)]
    log_neg_log_hs = [
        calc_log_neg_log_h(x_sizes, rs, q, log_epsilon)
        for x_sizes, rs in zip(x_sizes_all, rs_all)
    ]

    return np.array(fs), np.array(log_neg_log_hs)


def sample_vs(n_mc, n, l, q):
    """
    Return n_mc samples of random bidirectional connections.
    """

    return (np.random.rand(n_mc, 2*l, n) < q).astype(int)


def sample_vs_and_us(n_mc, n, l, q, r):

    # sample assoc->item cxns and get non-reciprocal scale factor d
    vs = sample_vs(n_mc, n, l, q)
    d = (1 - q*r) / (1 - q)

    # sample item->assoc cxns
    us = np.nan * np.zeros(vs.shape)

    us[vs.astype(bool)] = (np.random.rand(vs.sum()) < q*r).astype(int)
    us[~vs.astype(bool)] = (np.random.rand(vs.size - vs.sum()) < q*d).astype(int)

    return vs, us


def calc_xs_and_rs(v, u=None):
    """
    Calculate the intersection of each item's downstream neighbors with
    the maintained set of association units.
    """

    # calculate pairwise intersections
    if u is None: isctns = np.array([v[ctr] * v[ctr + 1] for ctr in range(0, len(v), 2)])
    else: isctns = np.array([u[ctr] * u[ctr + 1] for ctr in range(0, len(u), 2)])

    # calculate maintained set
    maintained = isctns.sum(axis=0).astype(bool).astype(int)

    # calculate xs
    if u is None: xs = np.array([(v[ctr] * maintained).astype(int) for ctr in range(len(v))])
    else: xs = np.array([(u[ctr] * maintained).astype(int) for ctr in range(len(v))])

    # calculate rs
    rs = np.array([
        np.sum(xs[ctr] * v[ctr + 1]) if ctr % 2 == 0 else np.sum(xs[ctr] * v[ctr-1])
        for ctr in range(len(v))
    ])

    return xs, rs


def calc_f(v, xs, rs):
    """
    Return 0 if there is interference among the provided set of cxns
    otherwise return 1.
    """

    if len(v) <= 2: return 1

    for ctr_0, (x, r) in enumerate(zip(xs, rs)):

        # pair to which the item belongs
        pair = (ctr_0, ctr_0 + 1) if ctr_0 % 2 == 0 else (ctr_0 - 1, ctr_0)

        remaining_units = [j for j in range(len(v)) if j not in pair]

        for ctr_1 in remaining_units:

            if (v[ctr_1] * x).sum() >= r: return 0

    return 1


def calc_log_neg_log_h(x_sizes, rs, q, log_epsilon):
    """
    Calculate the cumulative probability that getting less than r out of x_size
    connections with connection probability q.
    """

    # calc log survival function first
    log_sfs = stats.binom.logsf(rs - 1, x_sizes, q)

    # replace log_sf by log(-log_cdf) when log_sf is too big to approximate
    mask = log_sfs >= log_epsilon
    log_sfs[mask] = np.log(-stats.binom.logcdf(rs[mask] - 1, x_sizes[mask], q))

    # return the log of the sum of the sfs
    return aux.log_sum(log_sfs)
