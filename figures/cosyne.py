from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import stats

from capacity_analysis import recall_error_upper_bound_vs_item_number, log_max_items_with_low_recall_error
from plot import set_fontsize


def lif_example(
        SEED,
        TAU, V_REST, V_TH, V_RESET, V_MIN, T_RP,
        V_REV_SYN_EXC, TAU_SYN_EXC,
        TAU_FRE, K_WTA, WTA_TH, WTA_INH, NOISE,
        N_ITEMS, N_ASSOCS, P_CXN):
    """
    Demonstrate associative recall in an LIF network.
    """

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


def capacity_analysis(
        SEED,
        Q, L_EXAMPLE, NS_EXAMPLE, MS_EXAMPLE, ERR_MAX_EXAMPLE,
        NS, LS, ERR_MAX, M_TOL, N_SAMPLES_MC, N_TRIALS):
    """
    Perform an analysis of the capacity of an ideal network for recalling conjunctions.
    """

    np.random.seed(SEED)

    # plot example error rate vs. m

    ms_example = np.array(MS_EXAMPLE)

    error_bounds_example = {
        n: recall_error_upper_bound_vs_item_number(
            ms=ms_example, n=n, q=Q, l=L_EXAMPLE, n_samples_mc=N_SAMPLES_MC)
        for n in NS_EXAMPLE
    }

    # loop through ls and ns to calculate capacity

    capacity_means = {}
    capacity_sems = {}

    for l in LS:

        capacity_estimates = []

        for _ in range(N_TRIALS):

            capacity_estimates.append([
                log_max_items_with_low_recall_error(n, Q, l, ERR_MAX, N_SAMPLES_MC, M_TOL) for n in NS
            ])

        capacity_estimates = np.array(capacity_estimates) / np.log(10)

        capacity_means[l] = np.mean(capacity_estimates, axis=0)
        capacity_sems[l] = stats.sem(capacity_estimates, axis=0)

    # MAKE PLOTS

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    # example dependence on M

    handles_example = []

    for n in NS_EXAMPLE:

        handles_example.append(
            axs[0].loglog(
                ms_example, error_bounds_example[n], lw=2, label='N = {}'.format(n))[0])

    axs[0].axhline(ERR_MAX_EXAMPLE, color='k', lw=1)

    axs[0].set_xlabel('M')
    axs[0].set_ylabel('error upper bound')

    axs[0].legend(loc='best')

    # capacity vs. N

    handles = []

    for l in LS:

        handles.append(
            axs[1].errorbar(
                NS, capacity_means[l], yerr=capacity_sems[l], lw=2, label='L = {}'.format(l))[0])

    axs[1].set_xticks(NS)
    axs[1].set_xticklabels(NS, rotation=70)

    axs[1].set_xlabel('N')
    axs[1].set_ylabel('log10(max M)')
    axs[1].legend(loc='best')

    for ax in axs.flatten():

        set_fontsize(ax, 16)

    return fig
