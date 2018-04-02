
import numpy as np
from astropy.constants import R_sun, au
import astropy.units as u

r_sun_AU = float(R_sun / au)
AU_Mm = float(au / (1e6 * u.m))
'''
Helper methods for calculating white light signals. All of the functional forms
are taken from Howard & Tappin 2008.
'''
u = 0  # Limb darkening factor


def cossin(r):
    '''
    Returns cos omega and sin omega (see fig. 3)

    Parameters
    ----------
    r
        sun-blob distance normalised to sun radii
    '''
    sin = 1 / r
    cos = np.sqrt(1 - sin**2)
    return cos, sin


# A, B, C, D functions
def A(r):
    '''
    Parameters
    ----------
    r
        sun-blob distance in sun-radii
    '''
    cos, sin = cossin(r)
    return cos * sin**2


def B(r):
    '''
    Parameters
    ----------
    r
        sun-blob distance in sun-radii
    '''
    cos, sin = cossin(r)
    return -(1 / 8) * (1 -
                       3 * sin**2 -
                       ((cos**2 / sin) *
                        (1 + 3 * sin**2) *
                        np.log((1 + sin) / cos)))


def C(r):
    '''
    Parameters
    ----------
    r
        sun-blob distance in sun-radii
    '''
    cos, sin = cossin(r)
    return (4 / 3) - cos - (cos**3) / 3


def D(r):
    '''
    Parameters
    ----------
    r
        sun-blob distance in sun-radii
    '''
    cos, sin = cossin(r)
    return (1 / 8) * (5 +
                      sin**2 -
                      ((cos**2 / sin) *
                       (5 - sin**2) *
                       np.log((1 + sin) / cos)))


def tanphi(z, eps, r_obs):
    '''
    Observer - sun - electron angle as function of
    electron distance and sun - obeserver - electron angle

    Parameters
    ----------
    z
        distance along ray in AU
    eps
        observer-sun-blob angle in radians.
    r_obs
        sun-observer distance in AU
    '''
    return np.sin(eps) / ((r_obs / z) - np.cos(eps))


def sinphi(z, eps, r_obs):
    '''
    See above
    '''
    tan = tanphi(z, eps, r_obs)
    return tan / np.sqrt(1 + tan**2)


def r(z, eps, r_obs):
    '''
    Electron-sun distance in AU

    Parameters
    ----------
    z
        distance along ray in AU
    eps
        observer-sun-blob angle in radians.
    r_obs
        sun-observer distance in AU
    '''
    return np.abs(z * np.sin(eps) / sinphi(z, eps, r_obs))


def n_rsquared(r):
    '''
    Number density as a function of radial distance from Sun in AU
    '''
    return 1 / r**2


def nz_rsquared(z, eps, r_obs):
    '''
    Number density as a function of distance along observing ray.

    Parameters
    ----------
    z
        distance along ray in AU
    eps
        observer-sun-blob angle in radians.
    r_obs
        sun-observer distance in AU

    Notes
    -----
    Assumed 1/r^2 number density profile.
    '''
    rz = r(z, eps, r_obs)
    return n_rsquared(rz)


def G_t(z, eps, r_obs):
    rz = r(z, eps, r_obs)
    return (1 / z**2) * ((1 - u) * C(rz / r_sun_AU) + u * D(rz / r_sun_AU))


def G_p(z, eps, r_obs):
    rz = r(z, eps, r_obs)
    return (1 / z**2) * ((1 - u) * A(rz / r_sun_AU) + u * B(rz / r_sun_AU))


def G_tot(z, eps, r_obs):
    return 2 * G_t(z, eps, r_obs) - G_p(z, eps, r_obs)


def integrand_background(z, eps, r_obs):
    '''
    Intensity integrand for background 1/r^2 number desnity profile.

    Parameters
    ----------
    z
        distance along ray (float with dimensions of AU) in AU
    eps
        observer-sun-blob angle in radians.
    r_obs
        sun-observer distance in AU
    '''
    return nz_rsquared(z, eps, r_obs) * z**2 * G_tot(z, eps, r_obs)
