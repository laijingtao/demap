import numpy as np
import xarray as xr
import rioxarray
from typing import Union

from ._base import _speed_up, _XarrayAccessorBase
from .dem import _DEMAccessor
from .stream import _StreamAccessor


@xr.register_dataarray_accessor("demap")
class DemapDataarrayAccessor(_XarrayAccessorBase):

    def __init__(self, xrobj):
        super().__init__(xrobj)
        self._nodata = self._xrobj.rio.nodata

    @property
    def nodata(self):
        return self._nodata
    
    @nodata.setter
    def nodata(self, value):
        self._nodata = value

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
class DemapDatasetAccessor(_DEMAccessor, _StreamAccessor):
    
    pass
