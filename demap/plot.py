import numpy as np
from matplotlib.axes import Axes
from typing import Union

from .geogrid import GeoGrid
from .stream import StreamNetwork, Stream
from .swath import Swath


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
        i = s.ordered_nodes[:, 0]
        j = s.ordered_nodes[:, 1]

        x, y = s.rowcol_to_xy(i, j)

        ax.plot(x, y, **kwargs)

    return ax


def show_swath_loc(ax: Axes, swath: Swath, **kwargs):
    border_coords = swath.border_coords
    end_coords = swath.end_coords
    ax.plot(border_coords[:, 0], border_coords[:, 1], **kwargs)
    ax.plot(end_coords[:, 0], end_coords[:, 1], **kwargs)

    return ax
