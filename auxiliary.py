"""
Auxiliary functions.
"""
from __future__ import division, print_function
import numpy as np


def log_sum(log_xs):
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
        >>> log_sum(log_xs)
        -999.59239403555557
        >>> log_xs = np.array([1000,1001,1002])
        >>> np.log(np.sum(np.exp(log_xs)))
        inf
        >>> log_sum(log_xs)
        1002.4076059644444
    """
    # get largest element

    log_x_max = np.max(log_xs)

    # normalize log_x by subtracting log_x_max

    log_x_normed = log_xs - log_x_max

    # calculate sum of logarithms

    return log_x_max + np.log(np.sum(np.exp(log_x_normed)))
