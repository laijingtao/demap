import numpy as np
import xarray as xr
import rioxarray
from typing import Union

from ._base import _speed_up, _XarrayAccessorBase
from .dem import DEMAccessor
from .stream import StreamAccessor


@xr.register_dataarray_accessor("demap")
class DemapDataarrayAccessor(_XarrayAccessorBase):

    def __init__(self, xrobj):
        super().__init__(xrobj)
        #self._nodata = self._xrobj.rio.nodata

    @property
    def nodata(self):
        return self._xrobj.rio.nodata
    
    '''
    @nodata.setter
    def nodata(self, value):
        self._nodata = value
    '''

    def plotting_extent(self):
        """
        Returns an extent for for matplotlib's imshow (left, right, bottom, top)
        """
        rows, cols = self._xrobj.shape[0:2]
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (cols, rows)
        extent = (left, right, bottom, top)

        return extent


@xr.register_dataset_accessor("demap")
class DemapDatasetAccessor(_XarrayAccessorBase):
    
    def __init__(self, xrobj):
        super().__init__(xrobj)
        self.dem = DEMAccessor(xrobj)
        self.stream = StreamAccessor(xrobj)

    def __getattr__(self, name):
        # Delegate attribute access to self.dem if the attribute is not found in DemapDatasetAccessor
        if hasattr(self.dem, name):
            return getattr(self.dem, name)
        elif hasattr(self.stream, name):
            return getattr(self.stream, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")