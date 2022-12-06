import copy
import numpy as np

from ._base import is_verbose
from .geogrid import GeoGrid
from .stream import Stream, StreamNetwork
from .flow import (fill_depression, flow_direction, build_ordered_array, flow_accumulation)
from ._impl import _build_catchment_mask_impl


def process_dem(dem, **kwargs):
    if isinstance(dem, str):
        from .inout import load
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
        print("Warning: no stream_network is given, using exact x,y coordinates as catchment outlet.")
        outlet_i, outlet_j = receiver.xy_to_rowcol(x, y)
    else:
        outlet_i, outlet_j = stream_network.nearest_to_xy(x, y)

    mask = _build_catchment_mask_impl(outlet_i, outlet_j,
                                      receiver.dataarray.data, ordered_nodes)

    return mask

def extract_catchment_boundary(x, y, receiver: GeoGrid,
                               ordered_nodes: np.ndarray,
                               stream_network: StreamNetwork = None,
                               **kwargs):
    # TODO - does not work perfectly
    
    from scipy import spatial

    if is_verbose():
        print("Extracting catchment boundary ...")

    if stream_network is None:
        print("Warning: no stream_network is given, using exact x,y coordinates as catchment outlet.")
        outlet_i, outlet_j = receiver.xy_to_rowcol(x, y)
    else:
        outlet_i, outlet_j = stream_network.nearest_to_xy(x, y)

    mask = _build_catchment_mask_impl(outlet_i, outlet_j,
                                      receiver.dataarray.data, ordered_nodes)
    
    nrows, ncols, _ = receiver.dataarray.data.shape
    jj, ii = np.meshgrid(np.arange(ncols), np.arange(nrows))
    
    nodes_in = [ii[mask == 1], jj[mask == 1]]
    nodes_in = np.array(nodes_in).transpose()

    is_head = np.ones((nrows, ncols), dtype=bool)
    is_head[mask == 0] = False

    receiver_data = receiver.dataarray.data

    for k in range(len(nodes_in)):
        i, j = nodes_in[k]
        r_i, r_j = receiver_data[i, j]
        is_head[r_i, r_j]  = False

    mask_grad_x, mask_grad_y = np.gradient(mask)
    mask_grad = np.sqrt(np.power(mask_grad_x, 2) + np.power(mask_grad_y, 2))
    is_edge = np.zeros((nrows, ncols), dtype=bool)
    is_edge[mask_grad > 0] = True # boundary has non-zero gradient in mask
    is_edge[mask == 0] = False # boundary is in this catchment
    is_edge[is_head != 0] = False # boundary must be a channel head

    boundary_i = np.array(ii[is_edge == True])
    boundary_j = np.array(jj[is_edge == True])

    # sorting the boundary nodes
    # start from outlet, then add the closest node one by one.
    points = list(range(len(boundary_i)))
    in_list = np.zeros(len(points), dtype=bool)
    points_order = []
    k = 0
    points_order.append(points.pop(k))
    in_list[points_order[-1]] = True
    while len(points) > 0:
        curr_i = boundary_i[points_order[-1]]
        curr_j = boundary_j[points_order[-1]]
        dist = np.sqrt(np.power(boundary_i[in_list == False] - curr_i, 2) + np.power(boundary_j[in_list == False] - curr_j, 2))
        k = dist.argmin()
        points_order.append(points.pop(k))
        in_list[points_order[-1]] = True

    boundary_i = boundary_i[points_order]
    boundary_j = boundary_j[points_order]
    
    # make the boundary closed
    boundary_i = np.append(boundary_i, boundary_i[0])
    boundary_j = np.append(boundary_j, boundary_j[0])

    boundary_x, boundary_y = receiver.rowcol_to_xy(boundary_i, boundary_j)

    return boundary_x, boundary_y
