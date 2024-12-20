import numpy as np
import xarray as xr
import rasterio
from typing import Union

from ._base import _speed_up, _XarrayAccessorBase

class PlotAccessor(_XarrayAccessorBase):

    def get_extent(self):
        if hasattr(self._xrobj, 'hydro_order') and hasattr(self._xrobj, 'rows'):
            return self.get_stream_extent()
        else:
            return self.get_grid_extent()
        
    def get_stream_extent(self):
        x, y = self._xrobj.demap.stream_coords_xy

        return np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)

    def get_grid_extent(self):
        """
        Returns an extent for for matplotlib's imshow (left, right, bottom, top)
        """
        rows, cols = len(self._xrobj['y']), len(self._xrobj['x'])
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (cols, rows)
        extent = (left, right, bottom, top)

        return extent

    def get_hillshade(self, altitude=45, azimuth=135):
        """Calculate the hillshade of a suface.

        Parameters
        ----------
        dem : GeoGrid
            Input DEM
        altitude : float, optional
            The slope or angle of the illumination source above the horizon,
            ranges from 0 to 90 degrees, by default 45 degrees.
            0 degrees indicates that the sun is on the horizon.
        azimuth : float, optional
            The direction of the illumination source (in degrees).
            0 degrees is north and 90 degrees is east.
            Default is 135 (Southeast)

        Returns
        -------
        np.ndarray
            Hillshade of the suface.
            Ranges from 0 to 1, 0 being darkest and 1 being brightest
        """

        # https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst/how-hillshade-works.htm

        # TODO: add support for multidirectional hillshade

        if isinstance(self._xrobj, xr.DataArray):
            dem = self._xrobj
        elif isinstance(self._xrobj, xr.Dataset):
            dem = self._xrobj['dem']
        else:
            raise TypeError("Input DEM is not a xr.DataArray or xr.Dataset.")

        zenith_rad = (90.0 - altitude)/180.0 * np.pi

        azimuth_math = 360.0 - azimuth + 90
        if azimuth_math >= 360:
            azimuth_math -= 360
            
        azimuth_rad = azimuth_math/180.0 * np.pi

        dem_data = np.asarray(dem).astype(np.float32)
        dem_data[dem_data == dem.rio.nodata] = np.nan
        
        '''
        dx = self.dx
        dy = self.dy

        dzdx = np.zeros(dem_data.shape)
        dzdy = np.zeros(dem_data.shape)

        dzdx[1:-1, 1:-1] = ((dem_data[:-2, 2:]+2*dem_data[1:-1, 2:]+dem_data[2:, 2:]) - (dem_data[:-2, :-2]+2*dem_data[1:-1, :-2]+dem_data[2:, :-2])) / 8*dx
        dzdy[1:-1, 1:-1] = ((dem_data[2:, :-2]+2*dem_data[2:, 1:-1]+dem_data[2:, 2:]) - (dem_data[:-2, :-2]+2*dem_data[:-2, 1:-1]+dem_data[:-2, 2:])) / 8*dy
        '''

        dzdy, dzdx = np.gradient(dem_data, self.dy, self.dx)

        slope_rad = np.arctan(np.sqrt(np.power(dzdx, 2)+np.power(dzdy, 2))) 

        aspect_rad = np.zeros(dem_data.shape)
        aspect_rad[dzdx != 0] = np.arctan2(dzdy[dzdx != 0], -dzdx[dzdx != 0])
        aspect_rad[aspect_rad < 0] += 2*np.pi

        aspect_rad[np.logical_and(dzdx == 0, dzdy > 0)] = 0.5 * np.pi
        aspect_rad[np.logical_and(dzdx == 0, dzdy < 0)] = 1.5 * np.pi

        hillshade_value = ((np.cos(zenith_rad) * np.cos(slope_rad)) + \
            (np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)))

        hillshade_value = (hillshade_value + 1) / 2

        hillshade = xr.DataArray(hillshade_value, dims=dem.dims, coords=dem.coords)
        hillshade['spatial_ref'] = dem.spatial_ref

        return hillshade
    

    def plot_hillshade(self, ax, altitude=45, azimuth=135,
                       clip_ref: Union[xr.DataArray, xr.Dataset] = None,
                       clip_padding=0,
                       **kwargs):
        
        if clip_ref is not None:
            hillshade_dem = self._xrobj['dem'].demap.clip_by_ref(clip_ref=clip_ref, clip_padding=clip_padding)
        else:
            hillshade_dem = self._xrobj['dem']
        
        hillshade = hillshade_dem.demap.plot.get_hillshade(altitude, azimuth)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'Greys_r'
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
        if 'vmax' not in kwargs:
            kwargs['vmax'] = 1

        im = ax.imshow(hillshade, extent=hillshade.demap.plot.get_grid_extent(), **kwargs)

        return im

    def plot_stream(self, ax, **kwargs):

        streams = self._xrobj.demap.split_stream_network(mode='tributary')

        if 'color' not in kwargs:
            kwargs['color'] = 'C0'

        for st in streams[:]:
            x, y = st.demap.stream_coords_xy

            ax.plot(x, y, **kwargs)

    