import numpy as np

def auto_logspaced_bins(data):
    counts, bin_edges = np.histogram(np.log10(data[np.isfinite(data)]), bins='auto')
    bins = 10**bin_edges
    return counts, bins

def plot_log_hist(ax, data, **kwargs):
    bins = auto_logspaced_bins(data.dropna())
    ax.hist(data.dropna(), histtype='step', bins=bins, **kwargs)