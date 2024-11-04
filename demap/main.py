import numpy as np
import xarray as xr

from ._base import _speed_up
from .xr_accessor import DemapDataarrayAccessor, DemapDatasetAccessor

def process_dem(dem: xr.Dataset, **kwargs):
    DemapDatasetAccessor(dem).process_dem(**kwargs)
    return dem


def build_stream_network(dem: xr.Dataset, **kwargs):
    return DemapDatasetAccessor(dem).build_stream_network(**kwargs)

def extract_from_xy(stream_network: xr.Dataset, x, y, direction='up'):
    return DemapDatasetAccessor(stream_network).extract_from_xy(x, y, direction)

def split_stream_network(stream_network: xr.Dataset, mode='tributary'):
    return DemapDatasetAccessor(stream_network).split_stream_network(mode)

def stream_coords_rowcol(stream: xr.Dataset):
    return DemapDatasetAccessor(stream).stream_coords_rowcol

def stream_coords_xy(stream: xr.Dataset):
    return DemapDatasetAccessor(stream).stream_coords_xy

def stream_coords_latlon(stream: xr.Dataset):
    return DemapDatasetAccessor(stream).stream_coords_latlon


def project_stream_to_grid(dem_ds: xr.Dataset, stream_network_ds: xr.Dataset, var_name):

    stream_coords_rows, stream_coords_cols = stream_coords_rowcol(stream_network_ds)

    flow_dir_data = np.asarray(dem_ds['flow_dir'])
    ordered_pixels_data = np.asarray(dem_ds['ordered_pixels'])

    nearest_stream_node_row, nearest_stream_node_col = _assign_nearest_stream_node_impl(
        flow_dir_data,
        ordered_pixels_data,
        np.asarray(stream_coords_rows),
        np.asarray(stream_coords_cols)
    )

    gridded_value = _assign_stream_value_to_grid_impl(
        nearest_stream_node_row,
        nearest_stream_node_col,
        stream_coords_rows,
        stream_coords_cols,
        np.asarray(stream_network_ds[var_name], dtype=np.float32), # assume value is float type
        nodata=np.nan
    )

    dem_ds[var_name] = (('y', 'x'), gridded_value)

    return dem_ds[var_name]


@_speed_up
def _assign_nearest_stream_node_impl(flow_dir: np.ndarray,
                                     dem_ordered_pixels: np.ndarray,
                                     stream_coords_rows: np.ndarray,
                                     stream_coords_cols: np.ndarray,):
    

    nrow, ncol = len(flow_dir), len(flow_dir[0])

    nearest_stream_node_row = -np.ones((nrow, ncol), dtype=np.int32)
    nearest_stream_node_col = -np.ones((nrow, ncol), dtype=np.int32)

    #nearest_stream_node_row[stream_coords_rows, stream_coords_cols] = stream_coords_rows
    #nearest_stream_node_col[stream_coords_rows, stream_coords_cols] = stream_coords_cols

    for i, j in zip(stream_coords_rows, stream_coords_cols):
        nearest_stream_node_row[i, j] = i
        nearest_stream_node_col[i, j] = j

    """
    Demap's flow direction coding:
    |4|3|2|
    |5|0|1|
    |6|7|8|
    """
    di = np.array([0, 0, -1, -1, -1, 0, 1, 1, 1])
    dj = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

    for k in range(len(dem_ordered_pixels)-1, -1, -1):
        i, j = dem_ordered_pixels[k, 0], dem_ordered_pixels[k, 1]

        if nearest_stream_node_row[i, j] != -1:
            # already assigned, meaning this node is a stream node
            continue

        if flow_dir[i, j] == 0:
            # sink or outlet, nearest stream node is itself
            nearest_stream_node_row[i, j] = i
            nearest_stream_node_col[i, j] = j
            continue
        
        r_i = i + di[flow_dir[i, j]]
        r_j = j + dj[flow_dir[i, j]]

        if r_i >= 0 and r_i < nrow and r_j >= 0 and r_j < ncol:
            nearest_stream_node_row[i, j] = nearest_stream_node_row[r_i, r_j]
            nearest_stream_node_col[i, j] = nearest_stream_node_col[r_i, r_j]
        else:
            # receiver is out the bound, skip
            continue

    return nearest_stream_node_row, nearest_stream_node_col


def _assign_stream_value_to_grid_impl(nearest_stream_node_row: np.ndarray,
                                      nearest_stream_node_col: np.ndarray,
                                      stream_coords_rows: np.ndarray,
                                      stream_coords_cols: np.ndarray,
                                      value_along_stream: np.ndarray,
                                      nodata):
    # sink or outlet's nearest stream node is itself, so a nodata is needed for these nodes
    # this means that these nodes are not included in the gridded value, may cause some problem in future

    gridded_value = np.empty(nearest_stream_node_row.shape, dtype=value_along_stream.dtype)

    gridded_value[:] = nodata

    gridded_value[stream_coords_rows, stream_coords_cols] = value_along_stream

    gridded_value = gridded_value[nearest_stream_node_row, nearest_stream_node_col]

    return gridded_value