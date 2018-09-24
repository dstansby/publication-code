from datetime import datetime

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd

from heliopy.data import helios

ani_cut = 1.7
correl_cut = 0.8


def vzero(vs, bs):
    '''
    Takes a series of vs and bs, and works out the offset to apply to the vs to
    masimise the summed dot product squared of the vs and bs.

    See Sonnerup et. al. 1987 for details.
    '''
    bsum = (np.einsum('i, jk -> ijk', np.sum(bs**2, axis=1), np.identity(3)) -
            np.einsum('ij, ik -> ijk', bs, bs))
    matrix = np.mean(bsum, axis=0)
    vector = np.einsum('mab, mb -> ma', bsum, vs)
    vector = np.mean(vector, axis=0)
    return np.dot(np.linalg.inv(matrix), vector)


def correl(data, min_points=None):
    data = data.dropna()
    if data.shape[0] < min_points:
        return np.nan
    vs = data[['vp_x', 'vp_y', 'vp_z']].values
    bs = data[['va_x', 'va_y', 'va_z']].values
    vs = vs - vzero(vs, bs)
    bs = bs
    return (2 * np.mean(np.einsum('ij,ij->i', vs, bs)) /
            (np.mean(np.sum(vs**2, axis=1)) +
             np.mean(np.sum(bs**2, axis=1))))


def create_correls(data, period, min_points):
    # Resample data
    out_downsamp = []
    out_upsamp = []
    for i, upsamp in enumerate(probe_split(data)):
        if upsamp.empty:
            continue
        downsamp = upsamp.resample(period).mean()
        downsamp['Probe'] = str(i + 1)
        downsamp['correl'] = data[['vp_x', 'vp_y', 'vp_z', 'va_x', 'va_y', 'va_z']].resample(period).apply(correl)
        out_downsamp.append(downsamp)

        upsamp['correl'] = downsamp['correl'].reindex(upsamp.index, method='ffill')
        out_upsamp.append(upsamp)
    assert i == 1
    return pd.concat(out_downsamp, sort=False), pd.concat(out_upsamp, sort=False)


def pp_collision_freq(n, T_perp, T_par):
    c1 = 0.03
    leading_const = c1 * ((const.e.si**4) / (const.eps0**2 * const.m_p**(1/2) * const.k_B**(3/2)))
    out = leading_const * n / (T_perp * T_par**0.5)
    return out.to(1 / u.s)


def transit_time(r, v):
    return (r / v).to(u.s)


def collisional_age(r, v, n, T_perp, T_par):
    nu_pp = pp_collision_freq(n, T_perp, T_par)
    return (transit_time(r, v) * nu_pp).astype(float)


def hist2dwrapper(ax, xs, ys, xbins, ybins,
                  xlog=False, ylog=False, contour=False,
                  hist2dkwargs={}, contourkwargs={}):
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    cmap = hist2dkwargs.pop('cmap', 'Blues')

    # Plot 2D histogram
    counts, xbins, ybins, im = ax.hist2d(xs, ys,
                                         bins=(xbins, ybins),
                                         cmap=cmap, **hist2dkwargs);

    if contour:
        # Create contour bins and plot contour
        linewidths = contourkwargs.pop('linewidths', 0.5)
        colors = contourkwargs.pop('colors', 'k')
        X, Y = np.meshgrid((xbins[1:] + xbins[:-1]) / 2,
                           (ybins[1:] + ybins[:-1]) / 2)
        ax.contour(Y, X, counts, linewidths=0.5, colors='k', **contourkwargs)

    return im


def remove_solar_max(data):
    cutoff = datetime(1978, 1, 1)
    data = data.loc[data.index.get_level_values('Time') < cutoff]
    print('Number of points:', data.shape[0])
    return data
