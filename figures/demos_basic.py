from __future__ import division, print_function
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np

from figures import bulls_eye_layout, generate_visual_params, make_png_sequence
from network import make_basic_weight_matrix, Basic


def animation(
        SEED, TH, G_X, T_X, INH,
        N_ITEMS, N_ASSOCS, P_CXN, W_IA, W_AI,
        ITEMS, SAVE_DIR, PREFIX):
    """
    Generate animation demonstrating basic short-term memory storage
    in network.
    """
    np.random.seed(SEED)
    w = make_basic_weight_matrix(N_ITEMS, N_ASSOCS, P_CXN, W_IA, W_AI)
    ntwk = Basic(TH, G_X, T_X, N_ITEMS, w)

    # set up local and control inputs
    local_inputs = [None, None]
    controls = [None, None]

    # show two examples of single item memory
    for item in ITEMS[:2]:
        local_inputs.append((item, 1))
        controls.append(None)

        local_inputs.extend([None, None])
        controls.extend([None, None])
        local_inputs.append(None)
        controls.append(('item_blanket', .5))

        local_inputs.extend([None, None])
        controls.extend([None, None])

    local_inputs.extend([None, None, None, None, None])
    controls.extend([
        ('full_blanket', 4 * INH), ('full_blanket', 4 * INH),
        ('reset',), None, None])

    # show examples of associative memory
    item_pairs = [
        [ITEMS[:2], ITEMS[2:]],
        [[ITEMS[0], ITEMS[2]], [ITEMS[1], ITEMS[3]]]
    ]
    for item_pairs in item_pairs:
        for item_pair in item_pairs:

            local_inputs.append((list(item_pair), 1))
            controls.append(None)

            local_inputs.extend(2 * [None])
            controls.extend(2 * [('item_blanket', 4*INH)])

            local_inputs.extend(2 * [None])
            controls.extend(2 * [('item_blanket', INH)])

            local_inputs.append(([item_pair[0]], 1))
            controls.append(('item_blanket', INH))

            local_inputs.extend(2 * [None])
            controls.extend(2 * [('item_blanket', INH)])

            local_inputs.extend(2 * [None])
            controls.extend(2 * [('full_blanket', 4*INH)])

        local_inputs.extend([None, None, None])
        controls.extend([('reset',), None, None])

    # run network
    rs, xcs, vs = ntwk.run(local_inputs, controls)

    rs = rs[:-1]
    xcs = xcs[:-1]
    vs = vs[1:]

    # make animation
    pos = bulls_eye_layout(N_ITEMS, N_ASSOCS, 1.2, 1.3)
    node_colors, edge_colors = generate_visual_params(
        rs, xcs, vs, c_inactive=(0, 0, 0), c_active=(1, 1, 0), c_x=(.35, .35, 0),
        v_min=0, v_max=1.5, cmap='hot')

    size = N_ITEMS * [550] + N_ASSOCS * [250]
    make_png_sequence(
        SAVE_DIR, PREFIX, fig_size=(10, 10),
        pos=pos, w=w, colors=node_colors, edge_colors=edge_colors,
        edge_lw=3, size=size, lw=.1, x_lim=[-1.3, 1.3], y_lim=[-1.3, 1.3])

    return rs, xcs, vs
