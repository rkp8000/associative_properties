from __future__ import division, print_function
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os


def bulls_eye_layout(n_ring, n_center, r_ring, r_center):
    """
    Generate a set of positions for a network consisting of a structured outer ring and a
    random inner center
    :param n_ring: number of nodes in outer ring
    :param n_center: number of nodes in center
    :param r_ring: radius of ring
    :param r_center: enclosing radius of center
    :return: 2D array of positions
    """
    pos_ring = nx.drawing.circular_layout(range(n_ring), scale=r_ring)
    pos_ring = np.array([pos_ring[_] for _ in range(n_ring)])
    pos_ring -= pos_ring.mean(axis=0)
    pos_center = nx.drawing.random_layout(range(n_center), scale=r_center)
    pos_center = np.array([pos_center[_] for _ in range(n_center)])
    pos_center -= pos_center.mean(axis=0)

    return np.concatenate([pos_ring, pos_center])


def generate_visual_params(rs, xcs, vs,
        c_inactive=(0, 0, 0), c_active=(0, 1, 1), c_x=(0, .5, .5),
        v_min=0, v_max=1, cmap='hot'):
    """Generate a set of parameters to control a network animation."""
    rs_all = rs.copy()
    xcs_all = xcs.copy()
    vs_all = vs.copy()

    node_colors_all = []
    edge_colors_all = []

    for rs, xcs, vs in zip(rs_all, xcs_all, vs_all):
        # get node inner colors
        node_colors = [
            c_active if r == 1 else (c_x if xc > 0 else c_inactive)
            for r, xc in zip(rs, xcs)
        ]
        node_colors_all.append(node_colors)

        # get colormap indexes from vs
        cmap_idxs = (1 / (v_max - v_min)) * (vs - v_min)

        # get node edge colors
        edge_colors = getattr(cm, cmap)(cmap_idxs)
        edge_colors_all.append(edge_colors)

    return node_colors_all, edge_colors_all


def make_png_sequence(
        save_dir, prefix, fig_size, pos, w,
        colors, edge_colors, edge_lw, size, lw, x_lim, y_lim):
    """Generate a sequence of pngs to animate the activity of a network."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.set_aspect('equal')

    # plot network using networkx
    g = nx.from_numpy_matrix(w)
    nx.draw(g, ax=ax, pos=pos, node_size=0, width=lw)
    nodes = ax.scatter(pos[:, 0], pos[:, 1], c='k', s=size, lw=edge_lw)
    ax.fill_between(
        x_lim, [y_lim[0], y_lim[0]], [y_lim[1], y_lim[1]],
        color='gray', alpha=.3, zorder=-1)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # loop over all time steps
    file_names = []
    for t, (cs, ecs) in enumerate(zip(colors, edge_colors)):

        nodes.set_color(cs)
        nodes.set_edgecolors(ecs)

        # update network parameters
        file_name = '{}_{}.png'.format(prefix, t)
        path = os.path.join(save_dir, file_name)
        fig.savefig(path)

        file_names.append(file_name)

    return file_names
