from datetime import datetime
import multiprocessing

import numpy as np
import pandas as pd

import astropy.constants as const
import astropy.units as u
from heliopy.data import helios


def add_probe_index(df, probe):
    '''
    Add an extra "Probe" index level to df
    '''
    df['Probe'] = probe
    df = df.set_index('Probe', append=True, drop=True)
    return df


def temp2vth(temp, m):
    """
    Assumes velocities are floating point numbers in degrees Kelvin.
    """
    return np.sqrt(2 * const.k_B * temp * u.K /
                   (const.m_p * m)).to(u.km / u.s).value


def load_data():
    '''
    Load all the Helios corefit data
    '''
    starttime = datetime(1974, 1, 1, 0, 0, 0)
    endtime = datetime(1985, 1, 1, 0, 0, 0)

    # Import corefit data
    data = []
    for probe in ['1', '2']:
        corefit = helios.corefit(probe, starttime, endtime, try_download=False)
        corefit = add_probe_index(corefit, probe)
        data.append(corefit)
    corefit = pd.concat(data)
    print('Loaded corefit data')
    print('Start date:', corefit.index.min())
    print('End date:', corefit.index.max())
    print('Number of data points:', corefit.shape[0])
    print('Keys:')
    print(corefit.keys())
    return corefit


def remove_CMEs(data):
    # Import and process csv file
    cme_times = pd.read_csv('../../Helios/csv/helios_cme_catalog.csv')
    cme_times['Month'] = 1
    cme_times['Day'] = 1
    cme_times['Start'] = (pd.to_datetime(cme_times[['Year', 'Month', 'Day']]) +
                          pd.to_timedelta(cme_times['Start (DOY)'], unit='D'))
    cme_times['End'] = (cme_times['Start'] +
                        pd.to_timedelta(cme_times['Duration (h)'], unit='h'))
    cme_times = cme_times.drop(labels=['Year', 'Start (DOY)', 'Duration (h)',
                                       'Month', 'Day'], axis=1)

    # Remove data
    old_shape = data.shape
    print("Removing CMEs")
    index_times = data.index.get_level_values('Time')
    index_probes = data.index.get_level_values('Probe')
    to_remove = index_times < datetime(1900, 1, 1)

    for _, row in cme_times.iterrows():
        to_remove = (to_remove |
                     (index_times > row['Start']) &
                     (index_times < row['End']) &
                     (index_probes == str(row['Spacecraft'])))

    data = data.loc[~to_remove]
    print('Removed {} points'.format(old_shape[0] - data.shape[0]))
    return data


def p_mag(B):
    return (B * 1e-9)**2 / (2 * const.mu0.value)


def p_th(n, T):
    return n * 1e6 * const.k_B.value * T


def beta(n, T, B):
    return p_th(n, T) / p_mag(B)


def mass_flux(n, vr, r):
    return n * vr * r**2


def calculate_derived(data):
    data = data.copy()

    # Calculate plasma stuff
    data['|B|'] = np.linalg.norm(data[['Bx', 'By', 'Bz']].values, axis=1)
    data['|v|'] = np.linalg.norm(data[['vp_x', 'vp_y', 'vp_z']].values, axis=1)

    # Calculate pressures
    data['Tp_tot'] = (2 * data['Tp_perp'] + data['Tp_par']) / 3
    data['p_mag'] = p_mag(data['|B|'])
    data['p_th_par'] = p_th(data['n_p'], data['Tp_par'])
    data['p_th_tot'] = p_th(data['n_p'], data['Tp_tot'])
    data['Beta'] = (data['p_th_par'] / data['p_mag'])
    data['Beta_tot'] = (data['p_th_tot'] / data['p_mag'])
    data['Tani'] = data['Tp_perp'] / data['Tp_par']
    data['Tp_tot'] = (2 * data['Tp_perp'] + data['Tp_par']) / 3
    # Number density compensated for radial expansion
    data['n_p_norm'] = data['n_p'] * data['r_sun']**2
    data['mass_flux'] = mass_flux(data['n_p'].values * u.cm**-3,
                                     data['vp_x'].values * u.km / u.s,
                                     data['r_sun'].values * const.au).to(1 / u.s).value
    # Specific entropy
    data['Entropy'] = data['Tp_tot'] / data['n_p']**0.5
    for comp in ['x', 'y', 'z']:
        data['va_' + comp] = (data['B' + comp] * 1e-9 * 1e-3 /
                                 np.sqrt(const.m_p.value * data['n_p'] *
                                         1e6 * const.mu0.value))
    data['|va|'] = np.linalg.norm(data[['va_x', 'va_y', 'va_z']].values, axis=1)
    print('New keys:\n', data.keys())
    return data


def distance_filter(data, lower_r, upper_r):
    return data.loc[(data['r_sun'] > lower_r) & (data['r_sun'] < upper_r)]


def probe_split(data):
    index_probe = data.index.get_level_values('Probe')
    out = [data.loc[index_probe == '1'].copy(),
           data.loc[index_probe == '2'].copy()]

    def droplevel(data, level):
        data.index = data.index.droplevel(level)
        return data
    return [droplevel(df, 'Probe') for df in out]


def apply(x):
    data = x[0].sort_index()
    f, period, min_points = x[1]
    return data.resample(period, level='Time').apply(f, min_points=min_points)


def apply_downsampled_function(data, f, period, min_points, nproc=2):
    """
    Downsample *data*, and apply *f*(downsampled) to each individual packet
    of downsampled data.

    *f* must take a dataframe and return a scalar.

    Returns
    -------
    out_downsamp
        Downsampled data with f() applied
    out_upsamp
        Upsampled data with f() applied and forward filled
    """
    probes = ['1', '2']
    # Resample data
    split_data = probe_split(data)
    split_input = [(l, (f, period, min_points)) for l in split_data]
    assert len(split_input) == 2

    if nproc > 1:
        with multiprocessing.Pool(nproc) as p:
            out_downsamp = p.map(apply, split_input)
    else:
        out_downsamp = [apply(i) for i in split_input]

    out_downsamp = [pd.DataFrame(s) for s in out_downsamp]
    out_downsamp = [l for l in out_downsamp if not l.empty]
    split_input = [s for s in split_input if not s[0].empty]
    out_upsamp = [d.reindex(s[0].index, method='bfill') for d, s in
                  zip(out_downsamp, split_input)]

    def add_probe_index(df, probe):
        df['Probe'] = probe
        df = df.set_index('Probe', append=True, drop=True)
        return df

    out_upsamp = [add_probe_index(df, probe) for df, probe in zip(out_upsamp, probes)]
    out_downsamp = [add_probe_index(df, probe) for df, probe in zip(out_downsamp, probes)]

    return (pd.concat(out_downsamp, sort=False),
            pd.concat(out_upsamp, sort=False))
