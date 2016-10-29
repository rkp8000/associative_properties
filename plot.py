from __future__ import division, print_function
from matplotlib.pyplot import cm
import numpy as np


def set_fontsize(ax, fontsize):
    """Set fontsize of all axis text objects to specified value."""

    for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):

        txt.set_fontsize(fontsize)

    legend = ax.get_legend()

    if legend:
        for txt in legend.get_texts(): txt.set_fontsize(fontsize)


def get_n_colors(n, colormap='rainbow'):
    """
    Return a list of colors equally spaced over a color map.
    :param n: number of colors
    :param colormap: colormap to use
    :return: list of colors that can be passed directly to color argument of plotting
    """

    return getattr(cm, colormap)(np.linspace(0, 1, n))
