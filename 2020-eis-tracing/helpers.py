from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const

from heliopy import spice
from heliopy.data import ace

import numpy as np
from scipy.io import readsav
import sunpy.map
import sunpy.coordinates.frames as sunframes
import sunpy.io.fits
import pfsspy

from datetime import datetime


ace_dates = (datetime(2013, 1, 13), datetime(2013, 1, 31))
rss = 2.3

def get_B_map():
    """
    Return the synoptic map used for the PFSS model.
    
    Returns
    -------
    sunpy.map.GenericMap
    """
    [[br, header]] = sunpy.io.fits.read('maps/mrzqs130120t0304c2132_015.fits')
    br = br - np.mean(br)
    br = np.roll(br, header['CRVAL1'], axis=1)
    header['crval1'] = 0
    
    return sunpy.map.Map((br, header))

def get_gong_map():
    return read_gong_adapt('maps/adapt40311_03k012_201301200200_i00020100n1.fts', 0)


def read_gong_adapt(file, i):
    """
    Read in a single GONG-ADAPT map realisation.
    
    Parameters
    ----------
    file : path-like
        The .fits file to read.
    i : int
        The realisation number to read in. Must be between 0 and 11.
    
    Returns
    -------
    sunpy.map.GenericMap
    """
    file_data = sunpy.io.fits.read('maps/adapt40311_03k012_201301200200_i00020100n1.fts')
    [data, header] = file_data[0]
    data = data[i, ...]
    print(data.shape)
    data = np.roll(data, int(header['CRVAL1']), axis=1)
    m = sunpy.map.Map((data, header))
    # Fix some FITS entries...
    m.meta['date-obs'] = m.meta['maptime']
    m.meta['ctype1'] = 'CRLN-CEA'
    m.meta['ctype2'] = 'CRLT-CEA'
    m.meta.pop('naxis3')
    return m


def get_L1_flines():
    starttime = ace_dates[0]
    endtime = ace_dates[1]

    data = ace.swi_h3b(starttime, endtime)
    v_alpha = data.quantity('vHe2')
    
    B_map = get_B_map()

    flines, input, output = footpoints(data.index, v_alpha, B_map)
    return flines, input, output

def earth_trajectory(times):
    traj = spice.Trajectory('earth')
    traj.generate_positions(times, 'Sun', 'IAU_SUN')
    return traj


def project_to_ss(seeds, vsw, obstime):
    """
    Project a set of abitrary coordinates in the heliosphere on to the source
    surface.
    
    Parameters
    ----------
    seeds : astropy.coordinates.SkyCoord
        Seed points to be projected onto the source surface
    vsw : Quantity
        Solar wind velocity used to ballistically project backwards
    obstime : datetime
        Observation time of the map that the seeds will be traced through
        
    Returns
    -------
    seeds_ss : astropy.coordinates.SkyCoord
        Seed points projected on to the source surface.
    """
    seeds.representation_type = 'spherical'
    # Calculate the time it takes for the solar wind to travel radially to
    # the source surface
    dt =  (seeds.radius - rss * const.R_sun) / vsw
    # Construct the Carrington frame that existed when the plasma left th
    # source surface
    ss_frame = sunframes.HeliographicCarrington(obstime=seeds.obstime - dt)
    # Transform to this frame
    seeds_ss = seeds.transform_to(ss_frame)
    # Finally, set the radius to the source surface
    seeds_ss = SkyCoord(
        seeds_ss.lon,
        seeds_ss.lat,
        rss * 0.99 * const.R_sun,
        obstime = obstime,
        frame='heliographic_carrington')
    return seeds_ss


def trace_to_surface(seeds, synoptic_map, rss=rss, nrho=60):
    """
    seeds : astropy.coordinates.SkyCoord
        Seed coordinates on the source surface.
    synoptic_map : sunpy.map.GenericMap
        Input synoptic magnetogram.
    rss : scalar
        Source surface radius.
    nrho : int
        Number of grid points in the radial direciton
        of the PFSS model.
    
    Returns
    -------
    flines : pfsspy.flines.field_lines
        Traced field lines.
    pfss_input : pfsspy.Input
        PFSS input.
    pfss_output : pfsspy.Output
        PFSS output.
    """
    pfss_input = pfsspy.Input(synoptic_map, nrho, rss)
    print('Computing PFSS...')
    pfss_output = pfsspy.pfss(pfss_input)

    tracer = pfsspy.tracing.FortranTracer(max_steps=2000)
    print('Tracing field lines...')
    flines = tracer.trace(seeds, pfss_output)
    return flines, pfss_input, pfss_output


def footpoints(times, vsw, synoptic_map):
    """
    Given a set of times at L1, trace back to the Sun and return
    a set of magnetic field footpoints.
    
    Parameters
    ----------
    times : 
    vsw : Quantity
    synoptic_map : sunpy.map.GenericMap

    Returns
    -------
    flines : pfsspy.flines.field_lines
        Traced field lines.
    pfss_input : pfsspy.Input
        PFSS input.
    pfss_output : pfsspy.Output
        PFSS output.
    """
    traj = earth_trajectory(times)
    seeds = project_to_ss(traj.coords, vsw, synoptic_map.date).T
    flines, input, output = trace_to_surface(seeds, synoptic_map)
    return flines, input, output


def map_interp_carrington(m, fpoints):
    """
    Given a SunPy map, and a series of footpoints, interpolate
    the map data on to the footpoints.
    """
    fpoints = sunframes.HeliographicCarrington(fpoints.lon, fpoints.lat, fpoints.radius, obstime=m.date)
    pixels = m.world_to_pixel(fpoints)
    x, y = (pixels.x / u.pix).astype(int), (pixels.y / u.pix).astype(int)
    trace = m.data[y, x]
    # Remove points that are behind the Sun
    # trace = trace.astype(float)
    # trace[fpoints.distance > m.dsun] = np.nan
    return trace


def read_fipdata(fip_file):
    """
    Clean up the raw full sun scan data, and return a series
    of SunPy maps with coordinate system information.
    """
    eis_map = readsav(fip_file)
    new_maps = {}
    for key in eis_map:
        new_map = eis_map[key].copy()
        new_map[new_map == 0] = np.nan
        obstime = Time('2013-01-17')
        shape = new_map.shape
        scale = 0.5356 / 1938  # deg / pix
        header = sunpy.map.make_fitswcs_header(np.empty(shape),
                                               SkyCoord(0, 0, unit=u.deg,
                                                        frame="helioprojective",
                                                        obstime=obstime),
                                               scale=[scale, scale] * u.deg / u.pix)
        new_maps[key] = sunpy.map.Map(new_map, header)

    return new_maps


def parker_angle(v, r=const.au, lat=0*u.deg):
    omega_sun = 450 * u.km / u.s / const.au
    return np.arctan(omega_sun * r * np.cos(lat) / v).to(u.deg)


def carrington_header(time, shape_out):
    """
    Create the header for a Heliographic Carrington coordinate frame
    at a given *time*, with a given data *shape_in*.
    """
    return sunpy.map.make_fitswcs_header(
        np.zeros(shape_out),
        SkyCoord(0, 0, unit=u.deg,
                 frame="heliographic_carrington",
                 obstime=time),
        scale=[180 / shape_out[0],
               360 / shape_out[1]] * u.deg / u.pix,
        projection_code="CAR")


def wrap_field_line(coords):
    """
    Insert NaNs into a set of coordinates when they wrap, so that
    lines don't jump from 0 to 360 degrees on a plot.
    """
    lon = coords.lon.to_value(u.deg)
    # We want to wrap when lon gos from < 180 to > 180
    lon[lon > 180] -= 360
    jumps = np.where(np.abs(np.diff(lon) > 180))[0]
    if jumps.size > 0:
        jumps += 1
        lon = coords.lon
        lon = np.insert(lon, jumps, np.nan * u.deg)
        lat = coords.lat
        lat = np.insert(lat, jumps, np.nan * u.deg)
        r = coords.radius
        r = np.insert(r, jumps, np.nan * u.m)
        coords = SkyCoord(lon, lat, r, frame=coords.frame)
        
    return coords