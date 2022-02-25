import math
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
