from __future__ import division, print_function
import numpy as np
from scipy import stats


def test_sample_vs_with_small_and_large_q():

    from capacity_analysis import sample_vs

    np.random.seed(0)

    vs = sample_vs(n_mc=100, n=100, l=3, q=0.1)

    assert vs.shape == (100, 6, 100)
    assert vs.sum() < vs.size / 2

    vs = sample_vs(n_mc=100, n=100, l=3, q=0.9)

    assert vs.shape == (100, 6, 100)
    assert vs.sum() > vs.size / 2


def test_sample_vs_and_us_with_small_and_large_q_and_different_reciprocity_coefs():

    from capacity_analysis import sample_vs_and_us

    np.random.seed(0)

    # low density

    vs, us = sample_vs_and_us(n_mc=100, n=100, l=3, q=0.1, r=0)

    assert vs.shape == us.shape == (100, 6, 100)
    assert vs.sum() < vs.size / 2
    assert us.sum() < us.size / 2

    assert np.sum(vs * us) == 0

    vs, us = sample_vs_and_us(n_mc=100, n=100, l=3, q=0.1, r=10)

    assert vs.shape == us.shape == (100, 6, 100)
    assert vs.sum() < vs.size / 2
    assert us.sum() < us.size / 2

    assert np.sum(vs * us) == np.sum(vs) == np.sum(us)

    vs, us = sample_vs_and_us(n_mc=100, n=100, l=3, q=0.1, r=5)

    assert vs.shape == us.shape == (100, 6, 100)
    assert vs.sum() < vs.size / 2
    assert us.sum() < us.size / 2

    assert 0 < np.sum(vs * us) < np.sum(vs)

    # high density

    vs, us = sample_vs_and_us(n_mc=100, n=100, l=3, q=0.8, r=1.25)

    assert vs.shape == us.shape == (100, 6, 100)
    assert vs.sum() > vs.size / 2
    assert us.sum() > us.size / 2

    assert np.sum(vs * us) == np.sum(vs) == np.sum(us)

    vs, us = sample_vs_and_us(n_mc=100, n=100, l=3, q=0.8, r=1)

    assert vs.shape == us.shape == (100, 6, 100)
    assert vs.sum() > vs.size / 2
    assert us.sum() > us.size / 2

    assert 0 < np.sum(vs * us) < np.sum(vs)


def test_calc_xs_and_rs_works_on_symmetrical_examples():

    from capacity_analysis import calc_xs_and_rs

    v = np.array([
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    ])

    xs_correct = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ])

    rs_correct = np.array([2, 2, 2, 2, 4, 4])

    xs, rs = calc_xs_and_rs(v)

    np.testing.assert_array_equal(xs, xs_correct)
    np.testing.assert_array_equal(rs, rs_correct)


def test_calc_xs_and_rs_works_correctly_on_asymmetrical_examples():

    from capacity_analysis import calc_xs_and_rs

    u = np.array([
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    ])

    v = np.array([
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    ])

    xs_correct = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ])

    rs_correct = np.array([1, 2, 2, 2, 3, 2])

    xs, rs = calc_xs_and_rs(v, u)

    np.testing.assert_array_equal(xs, xs_correct)
    np.testing.assert_array_equal(rs, rs_correct)


def test_calc_f_works_on_examples():

    from capacity_analysis import calc_f

    v = np.array([
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    ])

    xs = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    ])

    # max overlap between each x and remaining v rows is:
    # [3, 5, 2, 3, 3, 4]

    # cases with no interference (f = 1)

    assert 1 == calc_f(v, xs, rs=np.array([3, 5, 2, 3, 3, 4]))
    assert 1 == calc_f(v, xs, rs=np.array([4, 5, 2, 3, 3, 4]))

    # cases with interference (f = 0)

    assert 0 == calc_f(v, xs, rs=np.array([3, 4, 2, 3, 3, 4]))
    assert 0 == calc_f(v, xs, rs=np.array([2, 5, 2, 3, 3, 4]))
    assert 0 == calc_f(v, xs, rs=np.array([3, 5, 2, 3, 2, 4]))
    assert 0 == calc_f(v, xs, rs=np.array([2, 5, 2, 2, 2, 1]))


def test_calc_log_neg_log_h_works_on_examples():

    from capacity_analysis import calc_log_neg_log_h


    # test in regime where no approximation is needed

    rs = np.array([5, 11, 19, 36, 79])
    x_sizes = np.array([8, 20, 35, 75, 145])
    q = 0.5
    log_epsilon = -3 * np.log(10)

    h_correct = np.prod([stats.binom.cdf(r-1, x_size, q) for r, x_size in zip(rs, x_sizes)])
    log_neg_log_h_correct = np.log(-np.log(h_correct))

    log_neg_log_h = calc_log_neg_log_h(x_sizes, rs, q, log_epsilon)

    assert round(log_neg_log_h, 7) == round(log_neg_log_h_correct, 7)

    # test in regime approximations are needed

    q = 0.1
    log_epsilon = -9 * np.log(10)

    h_correct = np.prod([stats.binom.cdf(r-1, x_size, q) for r, x_size in zip(rs, x_sizes)])
    log_neg_log_h_correct = np.log(-np.log(h_correct))

    log_neg_log_h = calc_log_neg_log_h(x_sizes, rs, q, log_epsilon)

    assert round(log_neg_log_h, 7) == round(log_neg_log_h_correct, 7)


def test_calc_log_mc_sum_works_on_examples():

    from auxiliary import log_sum
    from capacity_analysis import calc_log_mc_sum

    log_neg_log_hs = np.array([-3.1, -5, -7, -9, -20, -50])
    log_m = 5
    l = 3
    log_epsilon = -9 * np.log(10)

    # test case when fs are all 0
    fs = np.zeros(log_neg_log_hs.shape)

    log_mc_sum_correct = np.log(len(fs))
    log_mc_sum = calc_log_mc_sum(fs, log_neg_log_hs, log_m, l, log_epsilon)

    assert log_mc_sum == log_mc_sum_correct

    # test case when fs are all 1 and no approximations are needed
    log_neg_log_hs = np.array([-3.1, -5, -7, -4, -6.2, -8])
    fs = np.ones(log_neg_log_hs.shape)

    hs = np.exp(-np.exp(log_neg_log_hs))
    h_to_m_minus_2ls = hs**(np.exp(log_m) - 2*l)
    log_mc_sum_correct = np.log(np.sum(1 - h_to_m_minus_2ls))

    log_mc_sum = calc_log_mc_sum(fs, log_neg_log_hs, log_m, l, log_epsilon)

    assert round(log_mc_sum, 7) == round(log_mc_sum_correct, 7)

    # test case when fs are all 1 and approximations are always needed
    log_neg_log_hs = np.array([-30, -40, -50, -55, -60, -70])
    log_mc_sum_correct = log_sum(np.log(np.exp(log_m) - 2*l) + log_neg_log_hs)

    log_mc_sum = calc_log_mc_sum(fs, log_neg_log_hs, log_m, l, log_epsilon)

    assert round(log_mc_sum, 7) == round(log_mc_sum_correct, 7)
