import numpy as np
import xarray as xr
import copy
import math
import richdem

from .helpers import rowcol_to_xy, xy_to_rowcol, latlon_to_xy, xy_to_latlon
from ._base import INT
from .swath import Swath


class GeoGrid:
    """A grid with georeferencing and metadata"""

    def __init__(self, data, crs, transform, metadata, nodata=None):
        nrows, ncols = data.shape[0:2]

        self.dataarray = xr.DataArray(
            data,
            attrs = {
                'crs': crs,
                'transform': transform,
                'metadata': metadata,
            }
        )
        self.dataarray = self.dataarray.rename({'dim_0': 'rows', 'dim_1': 'cols'})
        self.dataarray = self.dataarray.assign_coords(rows = np.arange(nrows, dtype=INT))
        self.dataarray = self.dataarray.assign_coords(cols = np.arange(ncols, dtype=INT))

        if self.dataarray.attrs['crs'] is None or not self.dataarray.attrs['crs'].is_projected:
            #raise ValueError("DEMAP only works with projected coordinate system. "+\
            #    "Please convert your data projection using e.g. QGIS/ArcGIS/GDAL.")
            print("Warning: DEMAP only works with projected coordinate system. "+\
                "Please convert your data projection using e.g. QGIS/ArcGIS/GDAL.")

        if nodata is not None:
            self.dataarray.attrs['nodata'] = nodata
        else:
            try:
                self.dataarray.attrs['nodata'] = metadata.get('nodata', None)
            except:
                self.dataarray.attrs['nodata'] = None

        if self.dataarray.attrs['nodata'] is None:
            print('Warning: nodata value is None. If this is a DEM file, it may cause problems.')

    def __str__(self):
        return f'GeoGrid\n{self.dataarray.__str__()}'

    def __repr__(self):
        return f'GeoGrid\n{self.dataarray.__repr__()}'

    def to_rdarray(self):
        if self._valid_for_richdem():
            out_rd = richdem.rdarray(self.dataarray.data, no_data=self.dataarray.attrs['nodata'])
            out_rd.geotransform = self.dataarray.attrs['transform'].to_gdal()

            return out_rd

    def _valid_for_richdem(self):
        if (self.dataarray.attrs['nodata'] is None) or np.isnan(self.dataarray.attrs['nodata']):
            raise ValueError("Invalid nodata value for richdem: {}".format(self.dataarray.attrs['nodata']))
        if np.sum(np.isnan(self.dataarray.data)) > 0:
            raise ValueError("Invalid input for richdem: the grid contains nan value")

        return True

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.dataarray.attrs['transform'])

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.dataarray.attrs['transform'])

    def latlon_to_xy(self, lat, lon):
        return latlon_to_xy(lat, lon, self.dataarray.attrs['crs'])

    def xy_to_latlon(self, x, y):
        return xy_to_latlon(x, y, self.dataarray.attrs['crs'])

    def dx(self):
        return np.abs(self.dataarray.attrs['transform'][0])

    def dy(self):
        return np.abs(self.dataarray.attrs['transform'][4])

    def cartopy_crs(self):
        import cartopy.crs as ccrs
        return ccrs.epsg(self.dataarray.attrs['crs'].to_epsg())

    def for_plotting(self):
        data = copy.deepcopy(self.dataarray.data)
        data = data.astype(dtype=float)
        data[np.where(data == self.dataarray.attrs['nodata'])] = np.nan
        return data

    def plotting_extent(self):
        """
        Returns an extent for for matplotlib's imshow (left, right, bottom, top)
        """
        rows, cols = self.dataarray.data.shape[0:2]
        left, top = self.dataarray.attrs['transform'] * (0, 0)
        right, bottom = self.dataarray.attrs['transform'] * (cols, rows)
        extent = (left, right, bottom, top)

        return extent

    def line_profile(self, x1, y1, x2, y2, interpolation=True):
        #if interpolation:
        #    get_value_along_line = interp_along_line
        #else:
        #    get_value_along_line = nearest_along_line

        row1, col1 = self.xy_to_rowcol(x1, y1)
        row2, col2 = self.xy_to_rowcol(x2, y2)

        n = int(np.hypot(row1-row2, col1-col2))

        #z, i_list, j_list = get_value_along_line(self.data, row1, col1, row2, col2, n)
        
        i_list, j_list = np.linspace(row1, row2, n), np.linspace(col1, col2, n)
        
        if interpolation:
            from scipy import ndimage
            z = ndimage.map_coordinates(self.dataarray.data, np.vstack((i_list, j_list)))
        else:
            z = self.dataarray.data[i_list.astype(int), j_list.astype(int)]

        x_list, y_list = self.rowcol_to_xy(i_list, j_list)
        dx = x_list[1:] - x_list[:-1]
        dy = y_list[1:] - y_list[:-1]
        d_dist = np.hypot(dx, dy)
        dist = np.zeros(len(x_list))
        dist[1:] = np.cumsum(d_dist)

        return z, dist

    def swath_profile(self, x1, y1, x2, y2, swath_width=1e3, interpolation=True):

        perp_vector = np.cross([x1-x2, y1-y2, 0], [0, 0, 1])[:2]
        perp_vector = perp_vector / math.sqrt(perp_vector[0]**2 + perp_vector[1]**2)

        border_coords = np.zeros((5, 2))
        border_coords[0] = [x1, y1] + swath_width * 0.5 * perp_vector
        border_coords[1] = [x1, y1] - swath_width * 0.5 * perp_vector
        border_coords[2] = [x2, y2] - swath_width * 0.5 * perp_vector
        border_coords[3] = [x2, y2] + swath_width * 0.5 * perp_vector
        border_coords[4] = border_coords[0]  # make it a close box

        # Note: this whole part is ugly, need a more elegant way to do this.

        # upper and lower border points associated with x1, y1
        row1_1, col1_1 = self.xy_to_rowcol(border_coords[0, 0], border_coords[0, 1])
        row1_2, col1_2 = self.xy_to_rowcol(border_coords[1, 0], border_coords[1, 1])

        # upper and lower border points associated with x2, y2
        row2_1, col2_1 = self.xy_to_rowcol(border_coords[3, 0], border_coords[3, 1])
        row2_2, col2_2 = self.xy_to_rowcol(border_coords[2, 0], border_coords[2, 1])

        # number of sampling points along profile
        n_para = int(np.hypot(row1_1-row2_1, col1_1-col2_1))
        # number of sampling points perpendicular to profile
        n_perp = int(np.hypot(row1_1-row1_2, col1_1-col1_2))

        row_start_list = np.linspace(row1_1, row1_2, n_perp)
        col_start_list = np.linspace(col1_1, col1_2, n_perp)
        row_end_list = np.linspace(row2_1, row2_2, n_perp)
        col_end_list = np.linspace(col2_1, col2_2, n_perp)
        
        i_list = np.zeros((n_perp, n_para))
        j_list = np.zeros((n_perp, n_para))
        for k in range(n_perp):
            i_list[k] = np.linspace(row_start_list[k], row_end_list[k], n_para)
            j_list[k] = np.linspace(col_start_list[k], col_end_list[k], n_para)

        i_list = i_list.reshape(n_para*n_perp)
        j_list = j_list.reshape(n_para*n_perp)
        
        if interpolation:
            from scipy import ndimage
            z = ndimage.map_coordinates(self.dataarray.data, np.vstack((i_list, j_list)))
        else:
            z = self.dataarray.data[i_list.astype(int), j_list.astype(int)]

        z = z.reshape((n_perp, n_para))

        x_list, y_list = self.rowcol_to_xy(i_list[:n_para], j_list[:n_para])
        dx = x_list[1:] - x_list[:-1]
        dy = y_list[1:] - y_list[:-1]
        d_dist = np.hypot(dx, dy)
        dist = np.zeros(len(x_list))
        dist[1:] = np.cumsum(d_dist)

        end_coords = np.array([[x1, y1], [x2, y2]])

        return Swath(z, dist, end_coords, border_coords)


def interp_along_line(grid: np.ndarray, row1, col1, row2, col2, n=None):
    """
    Return a profile along line, using scipy.idimage.map-coordinates,
    which uses bicubic interpolation.
    """
    from scipy import ndimage

    if n is None:
        n = int(np.hypot(row1-row2, col1-col2))

    i_list, j_list = np.linspace(row1, row2, n), np.linspace(col1, col2, n)

    zi = ndimage.map_coordinates(grid, np.vstack((i_list, j_list)))

    return zi, i_list, j_list


def nearest_along_line(grid: np.ndarray, row1, col1, row2, col2, n):
    """
    Return a profile along line, using nearest neighbor sampling.
    """
    if n is None:
        n = int(np.hypot(row1-row2, col1-col2))

    i_list, j_list = np.linspace(row1, row2, n), np.linspace(col1, col2, n)

    zi = grid[i_list.astype(int), j_list.astype(int)]

    return zi, i_list, j_list
