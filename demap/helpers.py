import math
import numpy as np
from rasterio import Affine

from ._base import INT

def rowcol_to_xy(row, col, transform: Affine):
    """
    Returns geographic coordinates given GeoArray data (row, col) coordinates.

    This function will return the coordinates of the center of the pixel.
    """
    # offset for center
    if isinstance(row, list):
        row = np.array(row)
        col = np.array(col)
    row_off = 0.5
    col_off = 0.5
    return transform * (col+col_off, row+row_off)


def xy_to_rowcol(x, y, transform: Affine):
    """
    Returns GeoArray data (row, col) coordinates of the pixel that contains
    the given geographic coordinates (x, y)
    """
    col, row = ~transform * (x, y)
    col = np.floor(col).astype(dtype=INT)
    row = np.floor(row).astype(dtype=INT)
    return row, col


def xy_to_latlon(x, y, crs):
    import pyproj
    
    proj_crs = pyproj.crs.CRS.from_wkt(crs.wkt)
    p = pyproj.Proj(proj_crs)
    
    lon, lat = p(x, y, inverse=True)
    
    return lat, lon

def latlon_to_xy(lat, lon, crs):
    import pyproj
    
    proj_crs = pyproj.crs.CRS.from_wkt(crs.wkt)
    p = pyproj.Proj(proj_crs)
    
    x, y = p(lon, lat)
    
    return x, y


def transform_to_ndarray(transform: Affine) -> np.ndarray:
    a = transform
    b = np.zeros((3, 3))
    b[0, :] = a[:3]
    b[1, :] = a[3:6]
    b[2, :] = [0, 0, 1]
    return b


def distance_p2p(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
