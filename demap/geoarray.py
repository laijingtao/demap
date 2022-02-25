import numpy as np
import copy
import richdem

from .helpers import rowcol_to_xy, xy_to_rowcol


class GeoArray:
    """A array with georeferencing and metadata"""

    def __init__(self, data, crs, transform, metadata, nodata=None):
        self.data = data
        self.crs = crs
        if not self.crs.is_projected:
            raise ValueError("DEMAP only works with projected coordinate system.\
                Please convert your data projection using e.g. QGIS/ArcGIS/GDAL.")
        self.transform = transform
        self.metadata = metadata

        self.metadata['crs'] = self.crs
        self.metadata['transform'] = self.transform

        if nodata is not None:
            self.nodata = nodata
            self.metadata['nodata'] = nodata
        else:
            self.nodata = self.metadata.get('nodata', None)

        if self.nodata is None:
            print('Warning: nodata value is None.')

    def __str__(self):
        return f'GeoArray:\n{self.data}\nCRS: {self.crs}\nTransform: {self.transform}'

    def __repr__(self):
        return f'GeoArray({self.data})'

    def to_rdarray(self):
        out_rd = richdem.rdarray(self.data, no_data=self.nodata)
        out_rd.geotransform = self.transform.to_gdal()

        return out_rd

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def to_cartopy_crs(self):
        import cartopy.crs as ccrs
        return ccrs.epsg(self.crs.to_epsg())

    def for_plot(self):
        data = copy.deepcopy(self.data)
        data = data.astype(dtype=float)
        data[np.where(data == self.nodata)] = np.nan
        return data
