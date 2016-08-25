from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


def set_fontsize(ax, fontsize):
    """Set fontsize of all axis text objects to specified value."""

    for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        txt.set_fontsize(fontsize)

    legend = ax.get_legend()
    if legend:
        for txt in legend.get_texts():
            txt.set_fontsize(fontsize)