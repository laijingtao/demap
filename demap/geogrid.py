import numpy as np
import copy
import richdem

from .helpers import rowcol_to_xy, xy_to_rowcol


class GeoGrid:
    """A grid with georeferencing and metadata"""

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
        return f'GeoGrid:\n{self.data}\nCRS: {self.crs}\nTransform: {self.transform}'

    def __repr__(self):
        return f'GeoGrid({self.data})'

    def to_rdarray(self):
        out_rd = richdem.rdarray(self.data, no_data=self.nodata)
        out_rd.geotransform = self.transform.to_gdal()

        return out_rd

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def cartopy_crs(self):
        import cartopy.crs as ccrs
        return ccrs.epsg(self.crs.to_epsg())

    def for_plotting(self):
        data = copy.deepcopy(self.data)
        data = data.astype(dtype=float)
        data[np.where(data == self.nodata)] = np.nan
        return data

    def plotting_extent(self):
        """
        Returns an extent for for matplotlib's imshow (left, right, bottom, top)
        """
        rows, cols = self.data.shape[0:2]
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (cols, rows)
        extent = (left, right, bottom, top)

        return extent
