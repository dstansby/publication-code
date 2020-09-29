import numpy as np
from datetime import datetime
import astropy.units as u
import pandas as pd


class Stream:
    def __init__(self, start, end, b, name, color):
        self.start = start
        self.end = end
        self.name = name
        self.color = color
        self.b = b


stream_1 = Stream(datetime(2018, 11, 3), datetime(2018, 11, 5), 1.54 * u.G,
                  'S1', 'tab:purple')
stream_2 = Stream(datetime(2018, 11, 9, 12), datetime(2018, 11, 10), 1.54 * u.G,
                  'S2', 'tab:green')
stream_3 = Stream(datetime(2018, 11, 17), datetime(2018, 11, 19), 1.61 * u.G,
                  'S3', 'tab:orange')
streams = [stream_1, stream_2, stream_3]

figdir = '/Users/dstansby/Dropbox/Work/Papers/20massflux/figures'
# 244 points in inches
figwidth = 244 / 72


def create_sigma_cs(data, period, min_points):
    data = data.dropna()
    # Resample data
    downsamp = data.resample(period).mean()
    downsamp['correl'] = data[['vp_x', 'vp_y', 'vp_z', 'va_x', 'va_y', 'va_z']].groupby(pd.Grouper(freq=period)).apply(sigma_c)
    # print(data[['vp_x', 'vp_y', 'vp_z', 'va_x', 'va_y', 'va_z']])
    # downsamp['correl'] = data[['vp_x', 'vp_y', 'vp_z', 'va_x', 'va_y', 'va_z']].resample(period).apply(
    #     {['vp_x', 'vp_y', 'vp_z', 'va_x', 'va_y', 'va_z']: sigma_c})

    data['correl'] = downsamp['correl'].reindex(data.index, method='ffill')
    return downsamp, data


def sigma_c(data, min_points=1):
    if data.shape[0] < min_points:
        return np.nan
    vdotb, modvb = _sigma_c_terms(data)
    return (2 * np.mean(vdotb) /
            np.mean(modvb))


def _sigma_c_terms(data):
    vs = data[['vp_x', 'vp_y', 'vp_z']].values
    bs = data[['va_x', 'va_y', 'va_z']].values
    vs = vs - vzero(vs, bs)
    bs = bs
    vdotb = np.einsum('ij,ij->i', vs, bs)
    modvb = np.sum(vs**2, axis=1) + np.sum(bs**2, axis=1)
    return vdotb, modvb


def vzero(vs, bs):
    '''
    Takes a series of vs and bs, and works out the offset to apply to the vs to
    masimise the summed dot product squared of the vs and bs.

    This is the de-Hoffman-Teller frame, or equivalently the Alfvén phase velocity.

    See Sonnerup et. al. 1987 for details.
    '''
    bsum = (np.einsum('i, jk -> ijk', np.sum(bs**2, axis=1), np.identity(3)) -
            np.einsum('ij, ik -> ijk', bs, bs))
    matrix = np.mean(bsum, axis=0)
    vector = np.einsum('mab, mb -> ma', bsum, vs)
    vector = np.mean(vector, axis=0)
    return np.dot(np.linalg.inv(matrix), vector)
