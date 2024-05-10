import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from typing import Union
import rioxarray

from .geogrid import GeoGrid
from .stream import StreamNetwork, Stream
from .swath import Swath
from .xr_accessor import DemapDataarrayAccessor, DemapDatasetAccessor


def show_grid(ax: Axes, grid: GeoGrid, **kwargs):
    """
    Plot GeoGrid using matplotlib imshow.
    """
    if not isinstance(ax, Axes):
        raise TypeError("A matplotlib ax must be given")
    if not isinstance(grid, GeoGrid):
        raise TypeError("Wrong grid data type")

    data_plt = grid.for_plotting()
    extent = grid.plotting_extent()

    ax.imshow(data_plt, extent=extent, **kwargs)

    return ax


def show_stream(ax: Axes,
                streams: Union[np.ndarray, list, StreamNetwork, Stream],
                **kwargs):

    if not isinstance(ax, Axes):
        raise TypeError("A matplotlib ax must be given")

    if isinstance(streams, StreamNetwork):
        stream_list = streams.to_streams()
    elif isinstance(streams, Stream):
        stream_list = [streams]
    else:
        stream_list = streams

    for s in stream_list:
        i = s.dataset['rows'].data
        j = s.dataset['cols'].data

        x, y = s.rowcol_to_xy(i, j)

        ax.plot(x, y, **kwargs)

    return ax


def show_swath_loc(ax: Axes, swath: Swath, **kwargs):
    border_coords = swath.border_coords
    end_coords = swath.end_coords
    ax.plot(border_coords[:, 0], border_coords[:, 1], **kwargs)
    ax.plot(end_coords[:, 0], end_coords[:, 1], **kwargs)

    return ax


def get_hillshade(dem: Union[np.ndarray, xr.DataArray], altitude=45, azimuth=135):
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
    
    if not isinstance(dem, Union[np.ndarray, xr.DataArray]):
        raise ValueError("Wrong data type for dem.")

    zenith_rad = (90.0 - altitude)/180.0 * np.pi
    
    azimuth_math = 360.0 - azimuth + 90
    if azimuth_math >= 360:
        azimuth_math -= 360
        
    azimuth_rad = azimuth_math/180.0 * np.pi

    dem_data = np.asarray(dem)
    dem_data = dem_data.astype(float)
    dem_data[dem_data == dem.demap.nodata] = np.nan

    dx = dem.demap.dx
    dy = dem.demap.dy
    
    dzdx = np.zeros(dem_data.shape)
    dzdy = np.zeros(dem_data.shape)
    
    dzdx[1:-1, 1:-1] = ((dem_data[:-2, 2:]+2*dem_data[1:-1, 2:]+dem_data[2:, 2:]) - (dem_data[:-2, :-2]+2*dem_data[1:-1, :-2]+dem_data[2:, :-2])) / 8*dx
    dzdy[1:-1, 1:-1] = ((dem_data[2:, :-2]+2*dem_data[2:, 1:-1]+dem_data[2:, 2:]) - (dem_data[:-2, :-2]+2*dem_data[:-2, 1:-1]+dem_data[:-2, 2:])) / 8*dy
    
    slope_rad = np.arctan(np.sqrt(np.power(dzdx, 2)+np.power(dzdy, 2))) 
    
    aspect_rad = np.zeros(dem_data.shape)
    aspect_rad[dzdx != 0] = np.arctan2(dzdy[dzdx != 0], -dzdx[dzdx != 0])
    aspect_rad[aspect_rad < 0] += 2*np.pi
    
    aspect_rad[np.logical_and(dzdx == 0, dzdy > 0)] = 0.5 * np.pi
    aspect_rad[np.logical_and(dzdx == 0, dzdy < 0)] = 1.5 * np.pi

    hillshade_value = ((np.cos(zenith_rad) * np.cos(slope_rad)) + \
        (np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)))
    
    hillshade_value = (hillshade_value + 1) / 2
    
    return hillshade_value
