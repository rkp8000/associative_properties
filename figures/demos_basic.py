from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

import network
from plot import set_fontsize


def single_trial(
        SEED,
        N_ITEM_UNITS, N_MEMORY_UNITS, P_CXN,
        N_PAIRS):
    """
    Store and recall a single sequence of associations in a basic network.
    """
    
    # make network
    
    np.random.seed(SEED)
    ntwk = network.Basic(N_ITEM_UNITS, N_MEMORY_UNITS, P_CXN)
    
    # build stimulus
    
    drives = []
    
    # original stimulus presentation
    
    pairs = [(2*ii, 2*ii + 1) for ii in range(N_PAIRS)]
    
    for pair in pairs:
        
        drives.extend([
            {'type': 'item_drive', 'params': pair},
            {'type': 'item_inhibition'},
            {'type': 'item_inhibition'},
            ])
        
    for pair in pairs:
        
        drives.extend([
            {'type': 'item_drive', 'params': (pair[0],)},
            {'type': None},
            {'type': None},
            {'type': 'item_inhibition'},
            {'type': 'item_inhibition'},
            {'type': 'item_drive', 'params': (pair[1],)},
            {'type': None},
            {'type': None},
            {'type': 'item_inhibition'},
            {'type': 'item_inhibition'},
            ])
    
    # run network
    
    results = ntwk.run(drives, record=['atvn_item_idxs'])
    
    # display results
    
    fig, ax = plt.subplots(1, 1, facecolor='white', figsize=(15, 5), tight_layout=True)
    
    y_min = -1
    y_max = 2 * N_PAIRS
    
    for t, (drive, result) in enumerate(zip(drives, results['atvn_item_idxs'])):
        
        for item in result:
            
            ax.scatter(t, item, s=50, c='k', lw=0)
            
        if drive['type'] == 'item_drive':
            
            ax.fill_between(
                [t - 0.3, t + 0.3], [y_min, y_min], [y_max, y_max], color='r', alpha=0.2)
            
        elif drive['type'] == 'item_inhibition':
            
            ax.fill_between(
                [t - 0.3, t + 0.3], [y_min, y_min], [y_max, y_max], color='b', alpha=0.2)
    
    ax.set_xlim(-1, len(drives) + 1)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('time step')
    ax.set_ylabel('item unit')
    
    ax.set_yticks(range(0, 2 * N_PAIRS))
    
    set_fontsize(ax, 20)
    
    return fig


def single_network_correct_recall_probability(
        SEED,
        N_ITEM_UNITS, N_MEMORY_UNITS, P_CXN,
        N_PAIRS, N_TRIALS):
    """
    Test out the probability of correctly recalling a set of associations in a single instantiation of a network.
    """

    # make network

    np.random.seed(SEED)
    ntwk = network.Basic(N_ITEM_UNITS, N_MEMORY_UNITS, P_CXN)

    # loop over all numbers of pairings

    probs_correct = []
    
    for n_pairs in N_PAIRS:

        # loop over trials
        
        n_correct = 0
        
        for _ in range(N_TRIALS):

            # get random pairs

            pairs = np.random.choice(N_ITEM_UNITS, size=(n_pairs, 2), replace=False)

            # build storage phase stimulus

            drives = []

            for pair in pairs:

                drives.extend([
                    {'type': 'item_drive', 'params': pair},
                    {'type': 'item_inhibition'},
                    {'type': 'item_inhibition'},
                    ])

            # build recall phase stimulus

            for pair in pairs:

                drives.extend([
                    {'type': 'item_drive', 'params': (pair[0],)},
                    {'type': None},
                    {'type': None},
                    {'type': 'item_inhibition'},
                    {'type': 'item_inhibition'},
                    {'type': 'item_drive', 'params': (pair[1],)},
                    {'type': None},
                    {'type': None},
                    {'type': 'item_inhibition'},
                    {'type': 'item_inhibition'},
                    ])
                
            # run network
            
            results = ntwk.run(drives, record=['atvn_item_idxs'])
            
            # get times corresponding to network recall
            
            recall_ts = range(3 * n_pairs + 2, len(drives), 5)
            
            # check what percentage of the associations were correctly recalled
            
            assocs = [results['atvn_item_idxs'][t] for t in recall_ts]
            assocs_correct = [val for pair in zip(pairs, pairs) for val in pair]
            assocs_correct = [tuple(assoc) for assoc in assocs_correct]
            
            assocs = [sorted(assoc) for assoc in assocs]
            assocs_correct = [sorted(assoc) for assoc in assocs_correct]
            
            assert len(assocs) == len(assocs_correct)
            
            if assocs == assocs_correct:
                
                n_correct += 1
            
        probs_correct.append(n_correct / N_TRIALS)
        
    # make plot
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    ax.plot(N_PAIRS, probs_correct, color='k', lw=2)

    ax.set_ylim(0.5, 1.1)
    
    ax.set_xlabel('number of pairings')
    ax.set_ylabel('correct recall probability')
    
    set_fontsize(ax, 20)
    
    return fig