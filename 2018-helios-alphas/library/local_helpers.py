import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as const

from heliopy.data import helios
import helioshelp
import pathlib


def load_data(fitdir):
    fitdir = pathlib.Path(fitdir)
    dates = pd.read_csv('stream_times.csv', parse_dates=[1, 2])
    protons, alphas = [], []
    for _, row in dates.iterrows():
        # Import protons
        protons.append(helios.corefit(row['Probe'],
                                      row['Start'],
                                      row['End']).data)

        probe = row['Probe']
        protons[-1]['Probe'] = probe

        year = row['Start'].strftime('%Y')
        startdoy = int(row['Start'].strftime('%j'))
        enddoy = int(row['End'].strftime('%j'))
        # Import alphas
        this_alphas = []
        for doy in range(startdoy, enddoy + 1):
            this_alphas.append(pd.read_csv(
                fitdir /
                'helios{}'.format(probe) / '{}'.format(year) /
                'h{}_{}_{:03d}_alpha_fits.csv'.format(probe, year, doy),
                index_col=0, parse_dates=[0]))
            this_alphas[-1]['Probe'] = probe
        this_alphas = pd.concat(this_alphas)

        this_alphas = this_alphas[this_alphas.index > row['Start']]
        this_alphas = this_alphas[this_alphas.index < row['End']]

        print(this_alphas.index.min(), this_alphas.index.max(),
              protons[-1]['r_sun'].min(),
              protons[-1]['r_sun'].max())
        alphas.append(this_alphas)

    protons = pd.concat(protons)
    alphas = pd.concat(alphas)
    protons = helioshelp.calculate_derived(protons)

    def reindex(probe):
        this_p = protons[protons['Probe'] == probe]
        this_a = alphas[alphas['Probe'] == probe]
        return this_p.reindex(index=this_a.index)

    protons = pd.concat([reindex(probe) for probe in [1, 2]])
    alphas['r_sun'] = protons['r_sun']

    return protons, alphas


def par_energy_density(n, T):
    eps = 0.5 * (n.values * u.cm**-3) * const.k_B * (T.values * u.K)
    return eps.to(u.J / u.m**3)


def perp_energy_density(n, T):
    eps = (n.values * u.cm**-3) * const.k_B * (T.values * u.K)
    return eps.to(u.J / u.m**3)


def calculate_derived(protons, alphas):
    alphas['Tani'] = alphas['Ta_perp'] / alphas['Ta_par']
    protons['Tani'] = protons['Tp_perp'] / protons['Tp_par']
    alphas['Tp_ani'] = protons['Tani']

    alphas['Ta_tot'] = (2 * alphas['Ta_perp'] + alphas['Ta_par']) / 3
    protons['Tp_tot'] = (2 * protons['Tp_perp'] + protons['Tp_par']) / 3

    alphas['Ta/Tp_perp'] = alphas['Ta_perp'] / protons['Tp_perp']
    alphas['Ta/Tp_par'] = alphas['Ta_par'] / protons['Tp_par']
    alphas['Ta/Tp_tot'] = alphas['Ta_tot'] / protons['Tp_tot']

    protons['|B|'] = np.linalg.norm(protons[['Bx', 'By', 'Bz']].values, axis=1)
    protons['Beta'] = helioshelp.beta(protons['n_p'], protons['Tp_par'], protons['|B|'])
    alphas['Beta'] = helioshelp.beta(alphas['n_a'], alphas['Ta_par'], protons['|B|'])

    alphas['abundance'] = alphas['n_a'] / protons['n_p']
    alphas['vth_par'] = helioshelp.temp2vth(alphas['Ta_par'].values, 4)
    alphas['vth_perp'] = helioshelp.temp2vth(alphas['Ta_perp'].values, 4)
    alphas['|v|'] = np.linalg.norm(alphas[['va_x', 'va_y', 'va_z']].values, axis=1)

    for comp in ['x', 'y', 'z']:
        alphas['drift_' + comp] = alphas['va_'+ comp] - protons['vp_' + comp]
    alphas['|drift|'] = np.linalg.norm(alphas[['drift_x', 'drift_y', 'drift_z']].values, axis=1)

    for pkey in ['Tp_perp', 'Tp_par']:
        alphas[pkey] = protons[pkey]

    alphas['eps_p_par'] = par_energy_density(protons['n_p'], protons['Tp_par']).value
    alphas['eps_a_par'] = par_energy_density(alphas['n_a'], alphas['Ta_par']).value
    alphas['eps_p_perp'] = perp_energy_density(protons['n_p'], protons['Tp_perp']).value
    alphas['eps_a_perp'] = perp_energy_density(alphas['n_a'], alphas['Ta_perp']).value

    return protons, alphas
