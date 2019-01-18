import astropy.units as u
import numpy as np

omega_sun = 2.97e-6 / u.s


def CGL_Br(B0, r0, r):
    return B0 * (r0 / r)**2


def CGL_Bphi(theta, B0, r0, v, r):
    return -omega_sun * np.cos(theta) * (r / v) * CGL_Br(B0, r0, r)


def CGL_B(theta, B0, r0, v, r):
    return np.sqrt(CGL_Br(B0, r0, r)**2 + CGL_Bphi(theta, B0, r0, v, r)**2)


def CGL_tperp(T0, v, theta, r, r0):
    return T0 * CGL_B(theta, 1, r0, v, r)


def CGL_tpar(T0, v, theta, r, r0):
    return T0 * (r / r0)**-4 * CGL_B(theta, 1, r0, v, r)**-2
