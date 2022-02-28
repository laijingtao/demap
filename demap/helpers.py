import math
import numpy as np
from rasterio import Affine


def rowcol_to_xy(row, col, transform: Affine):
    """
    Returns geographic coordinates given GeoArray data (row, col) coordinates.

    This function will return the coordinates of the center of the pixel.
    """
    # offset for center
    row_off = 0.5
    col_off = 0.5
    return transform * (col+col_off, row+row_off)


def xy_to_rowcol(x, y, transform: Affine):
    """
    Returns GeoArray data (row, col) coordinates of the pixel that contains
    the given geographic coordinates (x, y)
    """
    col, row = ~transform * (x, y)
    col = math.floor(col)
    row = math.floor(row)
    return row, col


def transform_to_ndarray(transform: Affine) -> np.ndarray:
    a = transform
    b = np.zeros((3, 3))
    b[0, :] = a[:3]
    b[1, :] = a[3:6]
    b[2, :] = [0, 0, 1]
    return b


def distance_p2p(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


def interp_along_line(grid: np.ndarray, row1, col1, row2, col2):
    """
    Return a profile along line, using scipy.idimage.map-coordinates,
    which uses bicubic interpolation.
    """
    from scipy import ndimage

    length = int(np.hypot(row1-row2, col1-col2))
    i_list, j_list = np.linspace(row1, row2, length), np.linspace(col1, col2, length)

    zi = ndimage.map_coordinates(grid, np.vstack((i_list, j_list)))

    return zi, i_list, j_list

def nearest_along_line(grid: np.ndarray, row1, col1, row2, col2):
    """
    Return a profile along line, using nearest neighbor sampling.
    """
    length = int(np.hypot(row1-row2, col1-col2))
    i_list, j_list = np.linspace(row1, row2, length), np.linspace(col1, col2, length)

    zi = grid[i_list.astype(int), j_list.astype(int)]

    return zi, i_list, j_list
