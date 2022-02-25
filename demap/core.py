import copy
import numpy as np
import richdem

from ._base import is_verbose
from .geoarray import GeoArray
from .stream import Stream, StreamNetwork
from .helpers import xy_to_rowcol, distance_p2p
from ._impl import (_build_receiver_impl,
                    _build_ordered_array_impl,
                    _flow_accumulation_impl,
                    _build_catchment_mask_impl)


def process_dem(filename):
    from .io import load
    dem = load(filename)
    dem_filled = fill_depression(dem)
    receiver = flow_direction(dem_filled)
    ordered_nodes = build_ordered_array(receiver)
    drainage_area = flow_accumulation(receiver, ordered_nodes)
    stream_network = build_stream_network(receiver, drainage_area)
    #stream_list = stream_network.to_streams()

    result = {
        'dem': dem,
        'dem_filled': dem_filled,
        'receiver': receiver,
        'ordered_nodes': ordered_nodes,
        'drainage_area': drainage_area,
        'stream_network': stream_network,
    }

    return result


def fill_depression(dem: GeoArray):
    """Fill dipression using richDEM

    There are two ways to do this (see below) and Barnes2014 is the default.

    Epsilon filling will give a non-flat filled area:
    https://richdem.readthedocs.io/en/latest/depression_filling.html#epsilon-filling

    Barnes2014 also gives a not-flat filled area,
    and it produces more pleasing view of stream network
    https://richdem.readthedocs.io/en/latest/flat_resolution.html#
    """
    if is_verbose():
        print("Filling depressions ...")

    print("RichDEM fill depression output:")

    dem_rd = dem.to_rdarray()
    # richdem's filldepression does not work properly with int
    dem_rd = dem_rd.astype(dtype=float)

    # One way to do this is to use simple epsilon filling.
    #dem_rd_filled = richdem.FillDepressions(dem_rd, epsilon=True, in_place=False)

    # Use Barnes2014 filling method to produce more pleasing view of stream network
    # `Cells inappropriately raised above surrounding terrain = 0`
    # means 0 cells triggered this warning.
    dem_rd_filled = richdem.FillDepressions(dem_rd, epsilon=False, in_place=False)
    dem_rd_filled = richdem.ResolveFlats(dem_rd_filled, in_place=False)

    dem_rd_filled = np.array(dem_rd_filled)

    dem_filled = GeoArray(dem_rd_filled,
                          copy.deepcopy(dem.crs), copy.deepcopy(dem.transform),
                          copy.deepcopy(dem.metadata))

    return dem_filled


def flow_direction(dem: GeoArray):
    """Calculate flow direction using richDEM
    Note: only support D8 algorithm for now.

    Return:
        receiver: an GeoArray that stores receiver node information.
            Each element is a 1 x 2 array that denotes a pair of indices
            in the associated GeoArray.
    """
    if is_verbose():
        print("Calculating flow direction ...")

    flow_dir = _flow_dir_from_richdem(dem)

    receiver = build_receiver(flow_dir)

    return receiver


def _flow_dir_from_richdem(dem: GeoArray):
    """
    Return:
        flow_dir: a GeoArray contain the flow direction information.
            -1 -- nodata
            0 -- this node produces no flow, i.e., local sink
            1-8 -- flow direction coordinates

            Flow coordinates follow RichDEM's style:
            |2|3|4|
            |1|0|5|
            |8|7|6|
    """

    dem_rd = dem.to_rdarray()

    nodata_flow_dir = -1

    print("RichDEM flow direction output:")

    flow_prop = richdem.FlowProportions(dem=dem_rd, method='D8')

    flow_dir_data = np.zeros(dem.data.shape)

    flow_prop = np.array(flow_prop)
    node_info = flow_prop[:, :, 0]
    flow_info = flow_prop[:, :, 1:9]

    flow_dir_data = np.argmax(flow_info, axis=2) + 1
    flow_dir_data[np.where(node_info == -2)] = nodata_flow_dir
    flow_dir_data[np.where(node_info == -1)] = 0

    flow_dir = GeoArray(flow_dir_data,
                        copy.deepcopy(dem.crs), copy.deepcopy(dem.transform),
                        copy.deepcopy(dem.metadata), nodata=nodata_flow_dir)

    return flow_dir


def build_receiver(flow_dir: GeoArray):
    """Build receiver
    Return:
        receiver: an GeoArray that stores receiver node information.
            Each element is a 1 x 2 array that denotes a pair of indices
            in the associated GeoArray.
    """
    if is_verbose():
        print("Building receiver grid ...")

    receiver_data = _build_receiver_impl(flow_dir.data)
    receiver = GeoArray(receiver_data,
                        copy.deepcopy(flow_dir.crs), copy.deepcopy(flow_dir.transform),
                        copy.deepcopy(flow_dir.metadata), nodata=[-1, -1])

    return receiver


def build_ordered_array(receiver: GeoArray):
    """Build an ordered array.

    Return:
        ordered_nodes: in this array, upstream point is always in front of
            its downstream point. Each element is a 1 x 2 array that
            denotes a pair of indices in the associated GeoArray.

    """
    if is_verbose():
        print("Building ordered array ...")

    ordered_nodes = _build_ordered_array_impl(receiver.data)

    return ordered_nodes


def flow_accumulation(receiver: GeoArray, ordered_nodes: np.ndarray):
    """Flow accumulation
    """
    if is_verbose():
        print("Accumulating flow ...")

    dx = np.abs(receiver.transform[0])
    dy = np.abs(receiver.transform[4])
    cellsize = dx * dy
    drainage_area_data = _flow_accumulation_impl(receiver.data, ordered_nodes, cellsize)

    nodata = -1
    drainage_area_data[np.where(receiver.data[:, :, 0] == receiver.nodata[0])] = nodata

    drainage_area = GeoArray(drainage_area_data,
                             copy.deepcopy(receiver.crs), copy.deepcopy(receiver.transform),
                             copy.deepcopy(receiver.metadata), nodata=nodata)

    return drainage_area


def build_stream_network(receiver: GeoArray, drainage_area: GeoArray,
                         drainage_area_threshold=1e6):

    receiver_in_stream = copy.deepcopy(receiver)
    # all nodes with drainage_area smaller than the threshold are set as nodata
    receiver_in_stream.data[np.where(drainage_area.data < drainage_area_threshold)] = np.array(receiver.nodata)

    stream_network = StreamNetwork(receiver_in_stream)

    return stream_network


def get_value_along_stream(stream: Stream, grid: GeoArray):
    return stream.get_value(grid=grid)


def extract_catchment_mask(x, y, receiver: GeoArray,
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
        outlet_i, outlet_j = xy_to_rowcol(x, y, receiver.transform)
    else:
        outlet_i, outlet_j = stream_network.nearest_to_xy(x, y)

    mask = _build_catchment_mask_impl(outlet_i, outlet_j,
                                      receiver.data, ordered_nodes)

    return mask


def calculate_chi(drainage_area: GeoArray, stream_network: StreamNetwork,
                  ref_concavity=0.45, ref_drainage_area=1e6, **kwargs):
    """
    Calculate chi and save it in the given StreamNetwork.
    """
    if is_verbose():
        print("Calculating chi ...")
    drainage_area_data = drainage_area.data
    ordered_nodes = stream_network.ordered_nodes
    downstream = stream_network.downstream
    sn_rowcol_to_xy = stream_network.rowcol_to_xy

    chi = np.zeros(len(ordered_nodes))

    for k in range(len(ordered_nodes)-1, -1, -1):
        if downstream[k] != -1:
            i, j = ordered_nodes[k]
            r_i, r_j = ordered_nodes[downstream[k]]
            x1, y1 = sn_rowcol_to_xy(i, j)
            x2, y2 = sn_rowcol_to_xy(r_i, r_j)
            chi[k] = chi[downstream[k]] + (ref_drainage_area/drainage_area_data[i, j])**ref_concavity\
                * distance_p2p(x1, y1, x2, y2)

    stream_network.attrs['chi'] = chi


def calculate_chi_grid():
    # TODO
    pass
