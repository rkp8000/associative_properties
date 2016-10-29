from __future__ import division, print_function
from itertools import product as cproduct
import logging
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np
import threading

from capacity_analysis import max_items_low_error, log_upper_error_bound
from db import connect_and_make_session, check_tables_not_empty, prepare_logging
from db import _models
from network import LIFWithKWTAItems
from plot import get_n_colors, set_fontsize


def lif_example(
        SEED,
        TAU, V_REST, V_TH, V_RESET, V_MIN, T_RP,
        V_REV_SYN_EXC, TAU_SYN_EXC,
        TAU_FRE, K_WTA, WTA_TH, WTA_INH, NOISE,
        N_ITEMS, N_ASSOCS, P_CXN):
    """
    Demonstrate associative recall in an LIF network.
    """

    np.random.seed(SEED)

    cxn_ap = (np.random.rand(N_ITEMS, N_ASSOCS) < P_CXN).astype(float)

    W_PA = 1. * cxn_ap

    W_AP = 1. * cxn_ap.T

    W_AM = np.eye(N_ASSOCS)
    W_MA = 10 * np.eye(N_ASSOCS)
    W_MM = 15 * np.eye(N_ASSOCS)

    n_items = W_PA.shape[0]
    n_assocs = W_PA.shape[1]
    n_mems = W_MM.shape[0]

    n_neurons = n_items + n_assocs + n_mems

    ntwk = LIFWithKWTAItems(
        tau=TAU, v_rest=V_REST, v_th=V_TH, v_reset=V_RESET, v_min=V_MIN, t_rp=T_RP,
        v_rev_syn_exc=V_REV_SYN_EXC, tau_syn_exc=TAU_SYN_EXC,
        tau_fre=TAU_FRE, k_wta=K_WTA, wta_th=WTA_TH, wta_inh=WTA_INH, noise=NOISE,
        w_pa=W_PA, w_ap=W_AP, w_am=W_AM, w_ma=W_MA, w_mm=W_MM)

    # drive

    n_steps = 2000

    drives = {}
    drives['item'] = np.zeros((n_steps, n_items))

    drives['item'][50:120, 10] = 2
    drives['item'][50:120, 65] = 2
    drives['item'][250:320, 30] = 2
    drives['item'][250:320, 125] = 2
    drives['item'][450:520, 80] = 2
    drives['item'][450:520, 170] = 2
    drives['item'][800:850, 10] = 2
    drives['item'][1000:1050, 65] = 2
    drives['item'][1200:1250, 30] = 2
    drives['item'][1400:1450, 125] = 2
    drives['item'][1600:1650, 80] = 2
    drives['item'][1800:1850, 170] = 2

    drives_item_no_inh = drives['item'].copy()

    drives_item_inh = np.zeros((n_steps, n_items))

    drives_item_inh[150:200, :] = -10
    drives_item_inh[350:400, :] = -10
    drives_item_inh[550:600, :] = -10
    drives_item_inh[900:950, :] = -10
    drives_item_inh[1100:1150, :] = -10
    drives_item_inh[1300:1350, :] = -10
    drives_item_inh[1500:1550, :] = -10
    drives_item_inh[1700:1750, :] = -10
    drives_item_inh[1900:1950, :] = -10

    drives['item'] += drives_item_inh

    v_init = V_REST * np.ones((n_neurons,))
    g_init = np.zeros((n_neurons,))
    dt = 0.001

    results, ts = ntwk.run(drives, v_init, g_init, dt, record=['spikes', 'vs', 'fre_items'])

    ## MAKE PLOTS

    fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True, tight_layout=True)

    drive_times, drive_idxs = drives_item_no_inh.nonzero()
    inh_times, inh_idxs = drives_item_inh.nonzero()

    axs[0].scatter(drive_times * dt, drive_idxs, marker='*', s=50, c='r', lw=0)
    axs[0].scatter(inh_times * dt, inh_idxs, marker='*', s=50, c='b', lw=0)

    axs[0].set_ylim(-1, n_items)

    axs[0].set_ylabel('item neuron')

    axs[0].set_title('item neuron external drive')

    spike_times, spike_idxs = results['spikes'][:, :n_items].nonzero()

    axs[1].scatter(spike_times * dt, spike_idxs, marker='*', s=50, c='k', lw=0)
    axs[1].set_xlim(0, n_steps * dt)
    axs[1].set_ylim(-1, n_items)

    axs[1].set_ylabel('item neuron')
    axs[1].set_title('item neuron spikes')

    spike_times, spike_idxs = \
        results['spikes'][:, n_items + n_assocs:n_items + n_assocs + n_mems].nonzero()

    axs[2].scatter(spike_times * dt, spike_idxs, marker='*', s=4, c='k', lw=0)

    axs[2].set_ylim(-1, n_mems)

    axs[2].set_xlabel('time (s)')
    axs[2].set_ylabel('mem neuron')

    axs[2].set_title('mem neuron spikes')

    for ax in axs:

        set_fontsize(ax, 20)

    return fig


def _write_error_bound_vs_number_of_items(
        SEED, LOG_10_MS, N_TRIALS, N_MC, L, NS, Q, R, GROUP_NAME, LOG_FILE):
    """
    Calculate the expected error bound given network parameters and several
    numbers of item units.
    """

    np.random.seed(SEED)

    # set up database and logging
    session = connect_and_make_session('associative_properties')
    prepare_logging(LOG_FILE)

    log_ms = np.array(LOG_10_MS) * np.log(10)

    message = (
        '\n\nBeginning {} trials with NS = {}, N_MC = {}, \n'
        'L = {}, Q = {}, R = {}, GROUP_NAME= {}\n'
    ).format(N_TRIALS, NS, N_MC, L, Q, R, GROUP_NAME)
    logging.info(message)

    for tr_ctr, n in cproduct(range(N_TRIALS), NS):

        # log current n and trial number
        logging.info('Trial {}, N = {}'.format(tr_ctr, n))
        log_errors = log_upper_error_bound(log_ms=log_ms, n_mc=N_MC, n=n, l=L, q=Q, r=R)

        eb = _models.ErrorBound(
            log_10_ms=LOG_10_MS,
            n_mc=N_MC,
            n=n,
            l=L,
            q=Q,
            r=R,
            log_errors=log_errors,
            group_name=GROUP_NAME)

        session.add(eb)
        session.commit()

    logging.info('Complete.')


def write_error_bound_vs_number_of_items(**kwargs):
    """
    Call the previous function in a worker thread.
    """

    t = threading.Thread(target=_write_error_bound_vs_number_of_items, kwargs=kwargs)
    t.start()


def _write_item_capacity(
        SEED, MAX_LOG_10_ERROR, N_TRIALS, N_MC, LS, NS, QS, RS, GROUP_NAME, LOG_FILE):
    """
    Calculate the maximum number of items supported by a given network
    such that the recall error is below a lower bound.
    """

    np.random.seed(SEED)

    # set up database and logging
    session = connect_and_make_session('associative_properties')
    prepare_logging(LOG_FILE)

    message = (
        'Beginning {} trials with MAX_LOG_10_ERROR = {}, NS = {}, N_MC = {}, LS = {}, '
        'QS = {}, RS = {}, GROUP_NAME = {}'
    ).format(N_TRIALS, MAX_LOG_10_ERROR, NS, N_MC, LS, QS, RS, GROUP_NAME)
    logging.info(message)

    max_log_error = MAX_LOG_10_ERROR * np.log(10)

    for tr_ctr, n, l, q, r in cproduct(range(N_TRIALS), NS, LS, QS, RS):

        # log current values and trial number
        message = 'Trial {}, N = {}, L = {}, Q = {}, R = {}'.format(tr_ctr, n, l, q, r)
        logging.info(message)

        log_item_capacity = max_items_low_error(max_log_error, n_mc=N_MC, n=n, l=l, q=q, r=r)

        ic = _models.ItemCapacity(
            max_log_10_error=MAX_LOG_10_ERROR,
            n_mc=N_MC,
            n=n,
            l=l,
            q=q,
            r=r,
            log_item_capacity=log_item_capacity,
            group_name=GROUP_NAME)

        session.add(ic)
        session.commit()

    logging.info('Complete.')


def write_item_capacity(**kwargs):
    """
    Call the previous function in background thread.
    """

    t = threading.Thread(target=_write_item_capacity, kwargs=kwargs)
    t.start()


def error_bound_vs_number_of_items(GROUP_NAME, NS, L, Q, R):
    """
    Plot the error bound vs. number of items for several NS.
    """

    # preliminaries
    session = connect_and_make_session('associative_properties')
    check_tables_not_empty(session, _models.ErrorBound)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = get_n_colors(len(NS), colormap='hsv')
    handles = []

    for n, color in zip(NS, colors):

        records = session.query(_models.ErrorBound).filter(
            _models.ErrorBound.group_name == GROUP_NAME,
            _models.ErrorBound.n == n,
            _models.ErrorBound.l == L,
            _models.ErrorBound.q.between(0.99 * Q, 1.01 * Q),
            _models.ErrorBound.r.between(0.99 * R, 1.01 * R)).all()

        log_10_ms = np.array([record.log_10_ms for record in records])
        log_errors = np.array([record.log_errors for record in records])

        handles.append(
            ax.loglog(
                10 ** log_10_ms.mean(axis=0), np.exp(log_errors.mean(axis=0)),
                color=color, lw=2, label='N = {}'.format(n))[0])

    ax.set_xlabel('number of item units')
    ax.set_ylabel('expected incorrect\n recall probability')

    ax.legend(handles=handles, loc='best')

    set_fontsize(ax, 16)

    return fig


def item_capacity(GROUP_NAME, MAX_LOG_10_ERROR, LS, Q, R):
    """
    Plot the item capacity for a network for several LS and NS.
    """

    # preliminaries
    session = connect_and_make_session('associative_properties')
    check_tables_not_empty(session, _models.ItemCapacity)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = get_n_colors(len(LS), colormap='rainbow')
    handles = []

    m = _models.ItemCapacity

    for l, color in zip(LS, colors):

        records = session.query(_models.ItemCapacity).filter(
            m.group_name == GROUP_NAME,
            m.max_log_10_error.between(1.01 * MAX_LOG_10_ERROR, 0.99 * MAX_LOG_10_ERROR),
            m.l == l,
            m.q.between(0.99 * Q, 1.01 * Q),
            m.r.between(0.99 * R, 1.01 * R)).all()

        ns, log_cs = zip(*[(record.n, record.log_item_capacity) for record in records])

        handles.append(
            ax.semilogy(
                ns, np.exp(log_cs), 'o', c=color,
                markeredgecolor='none', label='L = {}'.format(l))[0])

    ax.set_xlabel('number of binding units')
    ax.set_ylabel('max number of items')

    ax.legend(handles=handles, loc='best')

    ax.set_title('MAX ERROR = 10^{}'.format(MAX_LOG_10_ERROR))

    set_fontsize(ax, 16)

    return fig

