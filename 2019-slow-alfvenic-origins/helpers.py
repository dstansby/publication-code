from datetime import datetime, timedelta
import multiprocessing
import pathlib

import numpy as np
import scipy.special
import pandas as pd
import matplotlib.dates as mdates

import astropy.constants as const
import astropy.units as u
from heliopy.data import helios


def p_mag(B):
    return (B * 1e-9)**2 / (2 * const.mu0.value)


def p_th(n, T):
    return n * 1e6 * const.k_B.value * T


def beta(n, T, B):
    return p_th(n, T) / p_mag(B)


def mass_flux(n, vr, r):
    return n * vr * r**2

def calculate_derived_proton(data):
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

    # data['nu_pp'] = collisional_freq(data['n_p'].values * u.cm**-3, data['Tp_par'].values * u.K, data['Tani'], 1, 1)
    print('New keys:\n', data.keys())
    return data


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


def _sigma_c_terms(data):
    vs = data[['vp_x', 'vp_y', 'vp_z']].values
    bs = data[['va_x', 'va_y', 'va_z']].values
    vs = vs - vzero(vs, bs)
    bs = bs
    vdotb = np.einsum('ij,ij->i', vs, bs)
    modvb = np.sum(vs**2, axis=1) + np.sum(bs**2, axis=1)
    return vdotb, modvb

def correl(data, min_points=0):
    data = data.dropna()
    if data.shape[0] < min_points:
        return np.nan
    vdotb, modvb = _sigma_c_terms(data)
    return (2 * np.mean(vdotb) /
            np.mean(modvb))
    
def correl_err(data, min_points=0):
    data = data.dropna()
    npoints = data.shape[0]
    if npoints < min_points:
        return np.nan
    correls = []
    for i in range(5):
        temp_data = data.sample(npoints * 8 // 10)
        correls.append(correl(temp_data))
    return np.std(correls)
    

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
        downsamp['correl_err'] = data[['vp_x', 'vp_y', 'vp_z', 'va_x', 'va_y', 'va_z']].resample(period).apply(correl_err)
        out_downsamp.append(downsamp)

        upsamp['correl'] = downsamp['correl'].reindex(upsamp.index, method='ffill')
        out_upsamp.append(upsamp)
    assert i == 1
    return pd.concat(out_downsamp, sort=False), pd.concat(out_upsamp, sort=False)


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


def temp2vth(temp, m=1):
    """
    Convert temperature to thermal speed.
    Assumes velocities are floating point numbers in degrees Kelvin.

    Parameters
    ----------
    m : particle mass, in multiples of proton mass
    """
    return np.sqrt(2 * const.k_B * temp * u.K /
                   (const.m_p * m)).to(u.km / u.s).value


def load_electrons(probe, starttime, endtime):
    '''
    Load electron data
    '''
    f = 'H1_fit_params_March75.npy'
    data = np.load(f, allow_pickle=True)
    data = data.tolist()
    alldata = []
    for d in data:
        d = data[d]
        ts = d['time']
        ts = pd.to_datetime(mdates.num2date(ts), utc=True).tz_localize(None)
        tcperp = d['core_Maxwell'][:, 1]
        kappa = d['halo_Kappa'][:, 3]
        alldata.append(pd.DataFrame({'Time': ts, 'Tc_perp': tcperp, 'kappa': kappa}))
    return pd.concat(alldata).set_index('Time')


output_dir = pathlib.Path('.')
def load_alphafit(probe, starttime, endtime, verbose=False):
    '''
    Load alpha data
    '''
    starttime_orig = starttime
    paramlist = []
    starttime_orig = starttime
    while starttime < endtime + timedelta(days=1):
        year = str(starttime.year)
        doy = starttime.strftime('%j')
        fname = (output_dir / 'alpha_data' /
                 'helios{}'.format(probe) / '{}'.format(year) /
                 'h{}_{}_{:03d}_alpha_fits.csv'.format(probe, year, int(doy)))
        if verbose:
            print(fname)
        try:
            params = pd.read_csv(fname, index_col=0, parse_dates=[0])
        except FileNotFoundError:
            starttime += timedelta(days=1)
            if verbose:
                print('{}/{} alphafit data not available'.format(year, doy))
            continue
        paramlist.append(params)
        starttime += timedelta(days=1)
    paramlist = pd.concat(paramlist, sort=True)
    paramlist = paramlist[(paramlist.index > starttime_orig) &
                          (paramlist.index < endtime)]
    return paramlist
