import math
import numpy as np

from .geogrid import GeoGrid
from .stream import Stream
from .swath import Swath


def valley_xsec_at_xy(grid: GeoGrid, stream: Stream, x, y, length, **kwargs):
    """Return the valley cross section at given geographic coordinates x, y

    Parameters
    ----------
    grid : demap.GeoGrid
    stream : demap.Stream
    x, y : float
        geographic coordinates, if it's not on the stream, a nearest node
        on the stream will be used.
    length : float
        length of the cross section

    Returns
    -------
    z : array-like
        value along the cross section from given grid
    dist : array-like
        distance along the cross section
    """

    smooth_range = kwargs.get('smooth_range', 1e3)

    row, col = stream.nearest_to_xy(x, y)
    dir_vector = stream.dir_vector_at_rowcol(row, col, smooth_range=smooth_range)
    dir_vector = np.append(dir_vector, 0)

    perp_vector = np.cross(dir_vector, [0, 0, 1])[:2]
    perp_vector = perp_vector / math.sqrt(perp_vector[0]**2 + perp_vector[1]**2)

    x1, y1 = np.array([x, y]) + length * 0.5 * perp_vector
    x2, y2 = np.array([x, y]) - length * 0.5 * perp_vector

    end_coords = np.array([[x1, y1], [x2, y2]])

    z, dist = grid.profile_along_line(x1, y1, x2, y2)

    return Swath(z, dist, end_coords)


def xsec_along_valley(grid: GeoGrid, stream: Stream, length, spacing, **kwargs):

    dist_up = stream.dist_up

    row, col = stream.ordered_nodes[0]
    x, y = stream.rowcol_to_xy(row, col)
    swath = valley_xsec_at_xy(grid, stream, x, y, length, **kwargs)

    swath_list = [swath]

    prev_k = 0
    k = 1
    while k < len(stream.ordered_nodes):
        if np.abs(dist_up[k] - dist_up[prev_k]) > spacing:
            row, col = stream.ordered_nodes[k]
            x, y = stream.rowcol_to_xy(row, col)
            swath = valley_xsec_at_xy(grid, stream, x, y, length, **kwargs)
            swath_list.append(swath)
            prev_k = k

        k += 1

    return swath_list
