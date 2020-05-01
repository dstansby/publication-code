"""
Functions for removing bad values in EIS maps.
"""
import numpy as np
from sunpy.map import Map

def clean_ne(log_ne_map):
    data = log_ne_map.data
    
    baddata = (data > 12) | (data < 7)
    data[baddata] = np.nan
    
    return Map((data, log_ne_map.meta))