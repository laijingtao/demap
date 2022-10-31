import copy
import numpy as np

from ._base import is_verbose
from .geogrid import GeoGrid
from .stream import Stream, StreamNetwork
from .flow import (fill_depression, flow_direction, build_ordered_array, flow_accumulation)
from ._impl import _build_catchment_mask_impl


def process_dem(dem, **kwargs):
    if isinstance(dem, str):
        from .io import load
        dem = load(dem)
    dem_cond = prepare_dem(dem, **kwargs)
    receiver = flow_direction(dem_cond)
    ordered_nodes = build_ordered_array(receiver)
    drainage_area = flow_accumulation(receiver, ordered_nodes)
    drainage_area_threshold = kwargs.get('drainage_area_threshold', 1e6)
    stream_network = build_stream_network(receiver, drainage_area,
        drainage_area_threshold=drainage_area_threshold)
    #stream_list = stream_network.to_streams()

    result = {
        'dem': dem,
        'dem_cond': dem_cond,
        'receiver': receiver,
        'ordered_nodes': ordered_nodes,
        'drainage_area': drainage_area,
        'stream_network': stream_network,
    }

    return result


def prepare_dem(dem: GeoGrid, **kwargs):
    dem = copy.deepcopy(dem)

    base_level = kwargs.get('base_level', None)
    if base_level is not None:
        dem.dataarray.data[dem.dataarray.data < base_level] = dem.dataarray.attrs['nodata']

    dem_filled = fill_depression(dem)

    return dem_filled


def build_stream_network(receiver: GeoGrid, drainage_area: GeoGrid,
                         drainage_area_threshold=1e6):

    if is_verbose():
        print("Building stream network ...")

    receiver_in_stream = copy.deepcopy(receiver)
    # all nodes with drainage_area smaller than the threshold are set as nodata
    receiver_in_stream.dataarray.data[drainage_area.dataarray.data < drainage_area_threshold] = np.array(receiver.dataarray.attrs['nodata'])

    stream_network = StreamNetwork(receiver_in_stream)

    return stream_network


def extract_catchment_mask(x, y, receiver: GeoGrid,
                           ordered_nodes: np.ndarray,
                           stream_network: StreamNetwork = None,
                           **kwargs):
    """
    Return a mask array showing the extent of the catchment for given outlet(x, y).
    if a stream network is given, the x, y will be changes to the nearest stream node.
    """

    if is_verbose():
        print("Extracting catchment mask ...")

    if stream_network is None:
        print("Warning: no stream_network is given")
        outlet_i, outlet_j = receiver.xy_to_rowcol(x, y)
    else:
        outlet_i, outlet_j = stream_network.nearest_to_xy(x, y)

    mask = _build_catchment_mask_impl(outlet_i, outlet_j,
                                      receiver.dataarray.data, ordered_nodes)

    return mask
