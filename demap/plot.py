import numpy as np
from matplotlib.axes import Axes
from typing import Union

from .geogrid import GeoGrid
from .stream import StreamNetwork, Stream


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
        i = s.coords[:, 0]
        j = s.coords[:, 1]

        x, y = s.rowcol_to_xy(i, j)

        ax.plot(x, y, 'b-', **kwargs)

    return ax
