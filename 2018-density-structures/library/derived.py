import astropy.units as u
import astropy.constants as const
import numpy as np


def p_mag(B):
    return B**2 / (2 * const.mu0)


def p_th(n, T):
    return n * const.k_B * T


def beta(n, T, B):
    return p_th(n, T) / p_mag(B)


def calc_derived(corefit):
    '''
    Method to calculate derived plasma values

    The minimum and maximum pressure are given by the minimum and
    maximum values of
    $$
        p_{tot} = n_{p}k_{B}\left ( T_{p} + \delta T_{\alpha} + \left (1 + 2\delta \right ) T_{e} \right ) +  p_{mag}
    $$
    where $\delta = n_{\alpha} / n_{p}$. From L to R the terms are:
        - Proton thermal pressure
        - Alpha thermal pressure
        - Electron thermal pressure
        - Magnetic pressure
    '''
    corefit['|B|'] = np.linalg.norm(corefit[['Bx', 'By', 'Bz']], axis=1)
    corefit['Tp'] = (2 * corefit['Tp_perp'] + corefit['Tp_par']) / 3
    corefit['Tani'] = corefit['Tp_perp'] / corefit['Tp_par']
    p_unit = u.Pa
    corefit['p_mag'] = p_mag(corefit['|B|'].values * u.T * 1e-9).to(p_unit).value
    corefit['p_th'] = p_th(corefit['n_p'].values * u.cm**-3, corefit['Tp'].values * u.K).to(p_unit).value
    corefit['Beta'] = corefit['p_th'] / corefit['p_mag']

    # Min/max electron temps
    Te_min = 0.2e6 * u.K
    Te_max = 0.4e6 * u.K
    # Min/max alpha temps
    Ta_min = 0.2e6 * u.K
    Ta_max = 1e6 * u.K
    # Min/max alpha/proton number density ratio
    delta_min = 0.01
    delta_max = 0.05

    def p_thermal(n_p, T_p, T_a, T_e, delta):
        p_p = p_th(n_p, T_p)
        p_a = p_th(n_p * delta, T_a)
        p_e = p_th((1 + 2 * delta) * n_p, T_e)
        return p_p + p_a + p_e

    # Min/max total pressures
    corefit['p_tot_min'] = (p_thermal(corefit['n_p'].values * u.cm**-3,
                                      corefit['Tp'].values * u.K,
                                      Ta_min, Te_min, delta_min).to(p_unit).value +
                            corefit['p_mag'])
    corefit['p_tot_max'] = (p_thermal(corefit['n_p'].values * u.cm**-3,
                                      corefit['Tp'].values * u.K,
                                      Ta_max, Te_max, delta_max).to(p_unit).value +
                            corefit['p_mag'])
    corefit['p_tot_ave'] = (corefit['p_tot_max'] + corefit['p_tot_min']) / 2
    return corefit
