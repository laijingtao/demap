import xarray as xr

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