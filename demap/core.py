import copy
import numpy as np
import richdem

from ._base import is_verbose
from .geogrid import GeoGrid
from .stream import Stream, StreamNetwork
from .helpers import xy_to_rowcol
from ._impl import (_build_receiver_impl,
                    _build_ordered_array_impl,
                    _flow_accumulation_impl,
                    _build_catchment_mask_impl)


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


def fill_depression(dem: GeoGrid):
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

    dem_rd = dem.to_rdarray()
    # richdem's filldepression does not work properly with int
    dem_rd = dem_rd.astype(dtype=float)

    print("RichDEM fill depression output:")
    # One way to do this is to use simple epsilon filling.
    #dem_rd_filled = richdem.FillDepressions(dem_rd, epsilon=True, in_place=False)

    # Use Barnes2014 filling method to produce more pleasing view of stream network
    # `Cells inappropriately raised above surrounding terrain = 0`
    # means 0 cells triggered this warning.
    dem_rd_filled = richdem.FillDepressions(dem_rd, epsilon=False, in_place=False)
    dem_rd_filled = richdem.ResolveFlats(dem_rd_filled, in_place=False)

    dem_rd_filled = np.array(dem_rd_filled)

    dem_filled = GeoGrid(dem_rd_filled,
                         copy.deepcopy(dem.dataarray.attrs['crs']),
                         copy.deepcopy(dem.dataarray.attrs['transform']),
                         copy.deepcopy(dem.dataarray.attrs['metadata']),
                         nodata=dem.dataarray.attrs['nodata'])

    return dem_filled


def flow_direction(dem: GeoGrid):
    """Calculate flow direction using richDEM
    Note: only support D8 algorithm for now.

    Return:
        receiver: an GeoGrid that stores receiver node information.
            Each element is a 1 x 2 array that denotes a pair of indices
            in the associated GeoGrid.
    """
    if is_verbose():
        print("Calculating flow direction ...")

    flow_dir = _flow_dir_from_richdem(dem)

    receiver = build_receiver(flow_dir)

    return receiver


def _flow_dir_from_richdem(dem: GeoGrid):
    """
    Return:
        flow_dir: a GeoGrid contain the flow direction information.
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

    flow_dir_data = np.zeros(dem.dataarray.data.shape)

    flow_prop = np.array(flow_prop)
    node_info = flow_prop[:, :, 0]
    flow_info = flow_prop[:, :, 1:9]

    flow_dir_data = np.argmax(flow_info, axis=2) + 1
    flow_dir_data[np.where(node_info == -2)] = nodata_flow_dir
    flow_dir_data[np.where(node_info == -1)] = 0

    flow_dir = GeoGrid(flow_dir_data,
                       copy.deepcopy(dem.dataarray.attrs['crs']),
                       copy.deepcopy(dem.dataarray.attrs['transform']),
                       copy.deepcopy(dem.dataarray.attrs['metadata']),
                       nodata=nodata_flow_dir)

    return flow_dir


def build_receiver(flow_dir: GeoGrid):
    """Build receiver
    Return:
        receiver: an GeoGrid that stores receiver node information.
            Each element is a 1 x 2 array that denotes a pair of coords
            in the associated GeoGrid.
            If receiver[i, j] = i, j, then this is an outlet or local sink
    """
    if is_verbose():
        print("Building receiver grid ...")

    receiver_data = _build_receiver_impl(flow_dir.dataarray.data)
    receiver = GeoGrid(receiver_data,
                       copy.deepcopy(flow_dir.dataarray.attrs['crs']),
                       copy.deepcopy(flow_dir.dataarray.attrs['transform']),
                       copy.deepcopy(flow_dir.dataarray.attrs['metadata']),
                       nodata=[-1, -1])

    return receiver


def build_ordered_array(receiver: GeoGrid):
    """Build an ordered array.

    Return:
        ordered_nodes: in this array, upstream point is always in front of
            its downstream point. Each element is a 1 x 2 array that
            denotes a pair of indices in the associated GeoGrid.

    """
    if is_verbose():
        print("Building ordered array ...")

    ordered_nodes = _build_ordered_array_impl(receiver.dataarray.data)

    return ordered_nodes


def flow_accumulation(receiver: GeoGrid, ordered_nodes: np.ndarray):
    """Flow accumulation
    """
    if is_verbose():
        print("Accumulating flow ...")

    dx = np.abs(receiver.dataarray.attrs['transform'][0])
    dy = np.abs(receiver.dataarray.attrs['transform'][4])
    cellsize = dx * dy
    drainage_area_data = _flow_accumulation_impl(receiver.dataarray.data, ordered_nodes, cellsize)

    nodata = -1
    drainage_area_data[np.where(receiver.dataarray.data[:, :, 0] == receiver.dataarray.attrs['nodata'][0])] = nodata

    drainage_area = GeoGrid(drainage_area_data,
                            copy.deepcopy(receiver.dataarray.attrs['crs']),
                            copy.deepcopy(receiver.dataarray.attrs['transform']),
                            copy.deepcopy(receiver.dataarray.attrs['metadata']),
                            nodata=nodata)

    return drainage_area


def build_stream_network(receiver: GeoGrid, drainage_area: GeoGrid,
                         drainage_area_threshold=1e6):

    if is_verbose():
        print("Building stream network ...")

    receiver_in_stream = copy.deepcopy(receiver)
    # all nodes with drainage_area smaller than the threshold are set as nodata
    receiver_in_stream.dataarray.data[drainage_area.dataarray.data < drainage_area_threshold] = np.array(receiver.dataarray.attrs['nodata'])

    stream_network = StreamNetwork(receiver_in_stream)

    return stream_network


def get_value_along_stream(stream: Stream, grid: GeoGrid):
    return stream.get_value(grid=grid)


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
