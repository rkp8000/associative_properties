"""
Tests of the auxiliary functions.
"""
from __future__ import division, print_function
import numpy as np


def test_log_sum_approximation_function():

    from auxiliary import log_sum

    log_xs = np.array([-1000, -1001, -1002])

    assert round(log_sum(log_xs) + 999.59239403555557, 7) == 0

    log_xs = np.array([1000,1001,1002])

    assert round(log_sum(log_xs) - 1002.4076059644444, 7) == 0