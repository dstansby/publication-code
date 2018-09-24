import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import numpy as np


def create_2d_1d_hist_axes(fig):
    """
    Create a 2D histogram axis, with adjoining axes at the top and right for
    1D histograms.
    """
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax2d = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_right)

    # Formatting
    for ax in (ax_top, ax_right):
        ax.set_frame_on(False)

    ax_right.yaxis.set_visible(False)
    ax_top.xaxis.set_visible(False)
    ax_right.xaxis.set_tick_params(size=0)
    ax_top.yaxis.set_tick_params(size=0)
    ax_right.set_xticklabels([])
    ax_top.set_yticklabels([])

    return ax2d, ax_top, ax_right


def auto_logspaced_bins(data):
    _, bin_edges = np.histogram(np.log10(data[np.isfinite(data)]), bins='auto')
    bins = np.logspace(bin_edges.min(), bin_edges.max(), bin_edges.size)
    return bins
