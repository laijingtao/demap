import numpy as np
import rasterio

from .helpers import rowcol_to_xy, xy_to_rowcol, latlon_to_xy, xy_to_latlon

try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

INT = np.int32

VERBOSE_MODE = True


def is_verbose():
    return VERBOSE_MODE


def set_verbose(verbose_mode):
    global VERBOSE_MODE
    VERBOSE_MODE = verbose_mode


def _speed_up(func):
    """A conditional decorator that use numba to speed up the function"""
    if USE_NUMBA:
        import numba
        return numba.njit(func)
    else:
        return func


class _XarrayAccessorBase:

    def __init__(self, xrobj):
        self._xrobj = xrobj
        #self._crs = self._xrobj.rio.crs
        #self._transform = self._xrobj.rio.transform()
    
    @property
    def crs(self):
        return self._xrobj.rio.crs
    
    '''
    @crs.setter
    def crs(self, value):
        self._crs = value
    '''

    @property
    def transform(self):
        return self._xrobj.rio.transform()
    
    '''
    @transform.setter
    def transform(self, value):
        self._transform = value
    '''

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def latlon_to_xy(self, lat, lon):
        return latlon_to_xy(lat, lon, self.crs)

    def xy_to_latlon(self, x, y):
        return xy_to_latlon(x, y, self.crs)

    '''
    @property
    def xy(self):
        return np.asarray(self._xrobj['x']), np.asarray(self._xrobj['y'])
    
    
    @property
    def rowcol(self):
        rows, cols = self.xy_to_rowcol(np.asarray(self._xrobj['x']), np.asarray(self._xrobj['y']))
        return rows, cols
    
    @property
    def latlon(self):
        lat, lon = self.xy_to_latlon(np.asarray(self._xrobj['x']), np.asarray(self._xrobj['y']))
        return lat, lon
    '''

    @property
    def dx(self):
        return np.abs(self.transform[0])

    @property
    def dy(self):
        return np.abs(self.transform[4])