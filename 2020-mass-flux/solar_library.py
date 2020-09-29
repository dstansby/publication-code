import astropy.units as u
import numpy as np
from skimage import measure
from scipy import ndimage
from scipy import signal

from sunpy.coordinates import frames
import sunpy.map
from sunpy.net import attrs as a
from sunpy.net import Fido
from sunpy.time import parse_time

from pubtools import solar as solartools


def get_gong_map(fname):
    """
    Return the synoptic map used for the PFSS model.
    
    Returns
    -------
    sunpy.map.GenericMap
    """
    [[br, header]] = sunpy.io.fits.read(fname)
    br = br - np.mean(br)
    br = np.roll(br, header['CRVAL1'], axis=1)
    header['CRVAL1'] = 0
    m = sunpy.map.Map((br, header))
    m.meta['date-obs'] = parse_time(m.meta['date-obs']).isot
    m = solartools.set_earth_obs_coord(m)
    return m
    

def get_AR_HMI_map(t):
    t = parse_time(t)
    if t < parse_time('2010-01-01'):
        series = 'mdi.fd_M_96m_lev182'
    else:
        series = "hmi.M_720s"
    result = Fido.search(a.Time(t, t),
                     a.jsoc.Series(series),
                     a.jsoc.Keys(["T_REC, CROTA2"]),
                     a.jsoc.Notify("jsoc@cadair.com"))
    hmi_map = Fido.fetch(result)
    m = sunpy.map.Map(hmi_map)
    m.meta.pop('CRDER1')
    m.meta.pop('CRDER2')
    return m