import numpy as np
import copy
import richdem

from .helpers import rowcol_to_xy, xy_to_rowcol, interp_along_line, nearest_along_line


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
        if self._valid_for_richdem():
            out_rd = richdem.rdarray(self.data, no_data=self.nodata)
            out_rd.geotransform = self.transform.to_gdal()

            return out_rd
    
    def _valid_for_richdem(self):
        if (self.nodata is None) or np.isnan(self.nodata):
            raise ValueError("Invalid nodata value for richdem: {}".format(self.nodata))
        if np.sum(np.isnan(self.data)) > 0:
            raise ValueError("Invalid input for richdem: the grid contains nan value")

        return True
        

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

    def profile_along_line(self, x1, y1, x2, y2, interpolation=True):
        row1, col1 = self.xy_to_rowcol(x1, y1)
        row2, col2 = self.xy_to_rowcol(x2, y2)
        
        if interpolation:
            z, i_list, j_list = interp_along_line(self.data, row1, col1, row2, col2)
        else:
            z, i_list, j_list = nearest_along_line(self.data, row1, col1, row2, col2)
        
        x_list, y_list = self.rowcol_to_xy(i_list, j_list)
        dx = x_list[1:] - x_list[:-1]
        dy = y_list[1:] - y_list[:-1]
        d_dist = np.hypot(dx, dy)
        dist = np.zeros(len(x_list))
        dist[1:] = np.cumsum(d_dist)

        return z, dist
