import copy
import math
import numpy as np
import rasterio as rio
import richdem

try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

INT = np.int32

VERBOSE = True


class GeoArray:
    """A array with georeferencing and metadata"""

    def __init__(self, data, crs, transform, metadata, nodata=None):
        self.data = data
        self.crs = crs
        if not self.crs.is_projected:
            raise ValueError("DEMAP only works with projected coordinate system.\
                Please convert your data projection using e.g. QGIS/ArcGIS/GDAL.")
        self.transform = transform
        self.metadata = metadata

        self.metadata['crs'] = self.crs
        self.metadata['transform'] = self.transform

        if nodata is not None:
            self.nodata = nodata
            self.metadata['nodata'] = nodata
        else:
            self.nodata = self.metadata.get('nodata', None)

        if self.nodata is None:
            print('Warning: nodata value is None.')

    def __str__(self):
        return f'GeoArray:\n{self.data}\nCRS: {self.crs}\nTransform: {self.transform}'

    def __repr__(self):
        return f'GeoArray({self.data})'

    def to_rdarray(self):
        out_rd = richdem.rdarray(self.data, no_data=self.nodata)
        out_rd.geotransform = self.transform.to_gdal()

        return out_rd

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def to_cartopy_crs(self):
        import cartopy.crs as ccrs
        return ccrs.epsg(self.crs.to_epsg())

    def for_plot(self):
        data = copy.deepcopy(self.data)
        data = data.astype(dtype=float)
        data[np.where(data == self.nodata)] = np.nan
        return data


class Stream:
    """A stream"""

    def __init__(self, coords):
        self.coords = np.array(coords, dtype=INT)  # n by 2 np.ndarray
        self.dist_up = None  # upstream distance
        self.attrs = {}

    def __repr__(self):
        return f'Stream({self.coords})'

    def get_upstream_distance(self, dx, dy):
        if len(self.coords) == 1:
            self.dist_up = np.array([0.])
            return self.dist_up

        dist_up = np.zeros(len(self.coords))
        dist_down = np.zeros(len(self.coords))

        i_list = self.coords[:, 0]
        j_list = self.coords[:, 1]

        d_i = np.abs(i_list[:-1] - i_list[1:])
        d_j = np.abs(j_list[:-1] - j_list[1:])
        d_dist = np.sqrt(np.power(d_i*dy, 2) + np.power(d_j*dx, 2))

        dist_down[1:] = np.cumsum(d_dist)
        dist_up = dist_down[-1] - dist_down

        self.dist_up = dist_up
        return self.dist_up

    def get_value(self, grid: GeoArray, name=None):
        i_list = self.coords[:, 0]
        j_list = self.coords[:, 1]

        if name is not None:
            self.attrs[name] = grid.data[i_list, j_list]

        return grid.data[i_list, j_list]


class StreamNetwork:
    """
    A StreamNetwork object saves the stream network using link information

    `ordered_nodes` saves the coordinates of all the nodes in this network,
    with upstream-to-downstream order.

    `downstream` and `upstream` save the link information. Both of them are
    saved as indexes in the `ordered_nodes`.

    `downstream` saves receiver info, `downstream[i]` is the receiver of `ordered_noded[i]`
    `upstream` saves donor info, `upstream[i]` is the donor of `ordered_noded[i]`
        `upstream` is a n x 9 np.ndarray, `upstream[i]` represents the number of
        donors plus 8 possilbe donor.

    if `downstream` is -1, then this is an outlet.
    if `
    """

    def __init__(self, receiver: GeoArray = None, **kwargs):
        if receiver is not None:
            self.build_from_receiver_array(receiver)
        elif all(k in kwargs for k in ('ordered_nodes', 'downstream', 'upstream')):
            self.ordered_nodes = kwargs['ordered_nodes']
            self.n_nodes = len(self.ordered_nodes)
            self._build_hashmap()
            self.downstream = kwargs['downstream']
            self.upstream = kwargs['upstream']
            self.crs = kwargs.get('crs', None)
            self.transform = kwargs.get('transform', None)
        else:
            print("Warning: incomplete input, an empty StreamNetwork was created")

    def build_from_receiver_array(self, receiver: GeoArray):
        if VERBOSE:
            print("Building stream network ...")

        self.transform = copy.deepcopy(receiver.transform)
        self.crs = copy.deepcopy(receiver.crs)

        self.ordered_nodes = build_ordered_array(receiver)
        self.n_nodes = len(self.ordered_nodes)
        self._build_hashmap()

        ordered_nodes = self.ordered_nodes
        i_list = ordered_nodes[:, 0]
        j_list = ordered_nodes[:, 1]
        # receiver_list follow the order of ordered_nodes
        receiver_list = receiver.data[i_list, j_list]

        downstream = -np.ones(self.n_nodes, dtype=INT)
        upstream = -np.ones((self.n_nodes, 9), dtype=INT)
        upstream[:, 0] = 0

        index_of = self.index_of
        for k in range(len(receiver_list)):
            from_idx = index_of(ordered_nodes[k, 0], ordered_nodes[k, 1])
            to_idx = index_of(receiver_list[k, 0], receiver_list[k, 1])

            if from_idx != to_idx:  # not outlet
                downstream[from_idx] = to_idx
                upstream[to_idx, 0] += 1  # number of upstream nodes
                upstream[to_idx, upstream[to_idx, 0]] = from_idx

        self.ordered_nodes = ordered_nodes
        self.downstream = downstream
        self.upstream = upstream

    def _build_hashmap(self):
        assert isinstance(self.ordered_nodes, np.ndarray), 'Missing ordered_nodes or wrong type'

        ordered_nodes = self.ordered_nodes
        nodes_hash = {}
        for k in range(len(ordered_nodes)):
            row, col = ordered_nodes[k]
            nodes_hash['{}_{}'.format(int(row), int(col))] = k

        self._nodes_hash = nodes_hash

    def index_of(self, row, col):
        return self._nodes_hash['{}_{}'.format(int(row), int(col))]

    def nearest_to_xy(self, x, y):
        """
        Return the stream node nearest to given x, y geographic coordinates.
        """
        row, col = self.xy_to_rowcol(x, y)
        d_i = np.abs(self.ordered_nodes[:, 0] - row)
        d_j = np.abs(self.ordered_nodes[:, 1] - col)
        dist = np.sqrt(np.power(d_i, 2) + np.power(d_j, 2))
        node = self.ordered_nodes[np.argmin(dist)]

        return node

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def to_streams(self, mode='all'):
        if VERBOSE:
            print("Converting stream network to streams ...")

        assert mode in ['all', 'tributary'], "Unknow mode, accepted modes are: \
            \'all\', \'tributary\'"

        ordered_nodes = self.ordered_nodes
        upstream = self.upstream
        downstream = self.downstream
        index_of = self.index_of

        # streams_idx[k] represents stream section starting from ordered_nodes[k]
        # recorded by index in ordered_nodes
        streams_idx = np.empty(self.n_nodes, dtype=object)

        for k in range(self.n_nodes-1, -1, -1):
            if downstream[k] == -1:
                streams_idx[k] = np.array([k])
            else:
                r_i, r_j = ordered_nodes[downstream[k]]
                streams_idx[k] = np.append(k, streams_idx[index_of(r_i, r_j)])

        # we only want stream start from the head
        streams_idx = streams_idx[np.where(upstream[:, 0] == 0)]

        # build streams from stream_idx
        streams = np.empty(len(streams_idx), dtype=object)
        dx = np.abs(self.transform[0])
        dy = np.abs(self.transform[4])
        for k in range(len(streams_idx)):
            streams[k] = Stream(coords=ordered_nodes[streams_idx[k]])
            streams[k].get_upstream_distance(dx, dy)

        # sort by length
        length_list = np.array([st.dist_up[0] for st in streams])
        sort_idx = np.argsort(length_list)[::-1]
        streams = streams[sort_idx]
        streams_idx = streams_idx[sort_idx]  # this is for split tributaries

        # split the stream network into tributaries
        if mode == 'tributary':
            in_streams = np.zeros(self.n_nodes, dtype=np.int8)

            for k in range(len(streams_idx)):
                not_in_streams = np.logical_not(in_streams[streams_idx[k]])
                streams[k].coords = streams[k].coords[np.where(not_in_streams)]
                streams[k].dist_up = streams[k].dist_up[np.where(not_in_streams)]

                # new coords, put them into network
                added_nodes_idx = streams_idx[k][np.where(not_in_streams)]
                in_streams[added_nodes_idx] = True

            # sort by length again
            length_list = np.array([st.dist_up[0] for st in streams])
            sort_idx = np.argsort(length_list)[::-1]
            streams = streams[sort_idx]

        return streams

    def to_shp(self):
        # TODO
        raise NotImplementedError

    def extract_from_xy(self, x, y, direction='up'):
        if VERBOSE:
            print("Extracting stream network ...")
        assert direction in ['up', 'down'], "Unknown direction, \'up\' or \'down\'"

        if direction == 'up':
            return self._extract_from_xy_up(x, y)
        elif direction == 'down':
            return self._extract_from_xy_down(x, y)

    def _extract_from_xy_down(self, x, y):
        index_of = self.index_of
        downstream = self.downstream
        ordered_nodes = self.ordered_nodes
        sub_mask = np.zeros(len(ordered_nodes), dtype=np.int8)

        i, j = self.nearest_to_xy(x, y)
        k = index_of(i, j)
        sub_mask[k] = True

        # remove all upstream nodes first
        new_upstream = copy.deepcopy(self.upstream)
        new_upstream[k] = -np.ones(9, dtype=INT)
        new_upstream[k, 0] = 0

        # go downstream
        while downstream[k] != -1:
            i, j = ordered_nodes[downstream[k]]
            k = index_of(i, j)
            sub_mask[k] = True

        new_ordered_nodes = copy.deepcopy(ordered_nodes[np.where(sub_mask)])
        new_downstream = copy.deepcopy(downstream[np.where(sub_mask)])
        new_upstream = new_upstream[np.where(sub_mask)]

        # update index in new_dowstream and new_upstream
        new_idx = -np.ones(len(ordered_nodes))
        new_idx[np.where(sub_mask)] = np.arange(len(new_ordered_nodes))
        # new_idx is essentially a function that convert old index to new index
        new_downstream[np.where(new_downstream >= 0)] = new_idx[new_downstream[np.where(new_downstream >= 0)]]
        new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)] = new_idx[new_upstream[:, 1:]
                                                                          [np.where(new_upstream[:, 1:] >= 0)]]

        sub_network = StreamNetwork(ordered_nodes=new_ordered_nodes,
                                    downstream=new_downstream,
                                    upstream=new_upstream)

        sub_network.crs = self.crs
        sub_network.transform = self.transform

        return sub_network

    def _extract_from_xy_up(self, x, y):
        index_of = self.index_of
        upstream = self.upstream
        ordered_nodes = self.ordered_nodes
        sub_mask = np.zeros(len(ordered_nodes), dtype=np.int8)

        i, j = self.nearest_to_xy(x, y)
        k = index_of(i, j)
        sub_mask[k] = True

        # remove all downstream nodes first
        new_downstream = copy.deepcopy(self.downstream)
        new_downstream[k] = -1

        # use a queue to build sub_mask
        from collections import deque
        q = deque([k], maxlen=len(ordered_nodes)+10)

        while len(q) > 0:
            k = q.popleft()
            for up_node in upstream[k, 1:upstream[k, 0]+1]:
                sub_mask[up_node] = True
                q.append(up_node)

        new_ordered_nodes = copy.deepcopy(ordered_nodes[np.where(sub_mask)])
        new_downstream = new_downstream[np.where(sub_mask)]
        new_upstream = copy.deepcopy(upstream[np.where(sub_mask)])

        # update index in new_dowstream and new_upstream
        new_idx = -np.ones(len(ordered_nodes))
        new_idx[np.where(sub_mask)] = np.arange(len(new_ordered_nodes))
        # new_idx is essentially a function that convert old index to new index
        new_downstream[np.where(new_downstream >= 0)] = \
            new_idx[new_downstream[np.where(new_downstream >= 0)]]
        new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)] = \
            new_idx[new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)]]

        sub_network = StreamNetwork(ordered_nodes=new_ordered_nodes,
                                    downstream=new_downstream,
                                    upstream=new_upstream)

        sub_network.crs = self.crs
        sub_network.transform = self.transform

        return sub_network


def rowcol_to_xy(row, col, transform: rio.Affine):
    """
    Returns geographic coordinates given GeoArray data (row, col) coordinates.

    This function will return the coordinates of the center of the pixel.
    """
    # offset for center
    row_off = 0.5
    col_off = 0.5
    return transform * (col+col_off, row+row_off)


def xy_to_rowcol(x, y, transform: rio.Affine):
    """
    Returns GeoArray data (row, col) coordinates of the pixel that contains
    the given geographic coordinates (x, y)
    """
    col, row = ~transform * (x, y)
    col = math.floor(col)
    row = math.floor(row)
    return row, col


def load(filename):
    """Read a geospatial data"""

    with rio.open(filename) as dataset:
        data = dataset.read(1)  # DEM data only have 1 band.
        crs = dataset.crs
        transform = dataset.meta['transform']
        metadata = dataset.meta

    outdata = GeoArray(data, crs, transform, metadata)

    return outdata


def fill_depression(dem: GeoArray):
    """Fill dipression using richDEM

    There are two ways to do this (see below) and Barnes2014 is the default.

    Epsilon filling will give a non-flat filled area:
    https://richdem.readthedocs.io/en/latest/depression_filling.html#epsilon-filling

    Barnes2014 also gives a not-flat filled area,
    and it produces more pleasing view of stream network
    https://richdem.readthedocs.io/en/latest/flat_resolution.html#
    """
    if VERBOSE:
        print("Filling depressions ...")
    dem_rd = dem.to_rdarray()
    # richdem's filldepression does not work properly with int
    dem_rd = dem_rd.astype(dtype=float)

    print("Fill depression - RichDEM output:")

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
    if VERBOSE:
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

    print("Flow direction - RichDEM output:")
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
    if VERBOSE:
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
    if VERBOSE:
        print("Building ordered array ...")

    ordered_nodes = _build_ordered_array_impl(receiver.data)

    return ordered_nodes


def flow_accumulation(receiver: GeoArray, ordered_nodes: np.ndarray):
    """Flow accumulation
    """
    if VERBOSE:
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

# old method, do not use this!


def extract_stream_network_from_receiver(
        receiver: GeoArray, drainage_area: GeoArray,
        drainage_area_threshold=1e6, mode='all'):
    assert mode in ['all', 'tributary'], "Unknow mode, accepted modes are: \'all\', \'tributary\'"
    ni, nj, _ = receiver.data.shape
    dx = np.abs(receiver.transform[0])
    dy = np.abs(receiver.transform[4])

    valid_receiver_data = copy.deepcopy(receiver.data)
    # all nodes with drainage_area smaller than the threshold are set as nodata
    valid_receiver_data[np.where(drainage_area.data < drainage_area_threshold)] = np.array(receiver.nodata)

    is_head = _is_head_impl(valid_receiver_data)

    stream_network = []
    for i in range(ni):
        for j in range(nj):
            if is_head[i, j]:
                stream_coords = _extract_stream_from_receiver_impl(i, j, valid_receiver_data)
                stream = Stream(coords=stream_coords)
                # all stream ends at a outlet, so we do dist_up here
                # the distance will be the distance to its outlet for each node
                stream.get_upstream_distance(dx, dy)
                stream_network.append(stream)
    stream_network = np.array(stream_network)

    # sort by length
    length_list = np.array([s.dist_up[0] for s in stream_network])
    sort_idx = np.argsort(length_list)[::-1]
    sort_idx = sort_idx.astype(dtype=int)
    stream_network = stream_network[sort_idx]

    if mode == 'tributary':
        # split the stream network into tributaries
        in_network = np.zeros((ni, nj))

        for stream in stream_network:
            i_list = stream.coords[:, 0]
            j_list = stream.coords[:, 1]
            in_network_mask = in_network[i_list, j_list]
            stream.coords = stream.coords[np.where(np.logical_not(in_network_mask))]
            stream.dist_up = stream.dist_up[np.where(np.logical_not(in_network_mask))]

            # new coords, put them into network
            i_list = stream.coords[:, 0]
            j_list = stream.coords[:, 1]
            in_network[i_list, j_list] = True

        # sort by length again
        length_list = np.array([s.dist_up[0] for s in stream_network])
        sort_idx = np.argsort(length_list)[::-1]
        sort_idx = sort_idx.astype(dtype=int)
        stream_network = stream_network[sort_idx]

    return stream_network


def get_value_along_stream(stream: Stream, grid: GeoArray):
    return stream.get_value(grid=grid)


# FIXME - align x y to nearest stream node.
def extract_catchment_mask(x, y, receiver: GeoArray, ordered_nodes: np.ndarray):
    """
    Return a mask array showing the extent of the catchment for given outlet(x, y).
    """
    outlet_i, outlet_j = xy_to_rowcol(x, y, receiver.transform)

    mask = _build_catchment_mask_impl(outlet_i, outlet_j,
                                      receiver.data, ordered_nodes)

    return mask


def clip_mask():
    # TODO
    raise NotImplementedError


def process_dem(filename):
    dem = load(filename)
    dem_filled = fill_depression(dem)
    receiver = flow_direction(dem_filled)
    ordered_nodes = build_ordered_array(receiver)
    drainage_area = flow_accumulation(receiver, ordered_nodes)
    stream_network = build_stream_network(receiver, drainage_area)
    stream_list = stream_network.to_streams()

    result = {
        'dem': dem,
        'dem_filled': dem_filled,
        'receiver': receiver,
        'ordered_nodes': ordered_nodes,
        'drainage_area': drainage_area,
        'stream_network': stream_network,
        'stream_list': stream_list
    }

    return result


# ============================================================
# Implementation. These methods can use `numba` to speed up.
# ============================================================
# Currently all these functions set default nodata [-1, -1] for receiver.
# Plan to enable a way to pass a nodata value for these funcs in the future.

def _speed_up(func):
    """A conditional decorator that use numba to speed up the function"""
    if USE_NUMBA:
        return numba.njit(func)
    else:
        return func


@_speed_up
def _build_receiver_impl(flow_dir: np.ndarray):
    """Implementation of build_receiver"""
    di = [0, 0, -1, -1, -1, 0, 1, 1, 1]
    dj = [0, -1, -1, 0, 1, 1, 1, 0, -1]

    ni, nj = flow_dir.shape
    receiver = np.zeros((ni, nj, 2), dtype=INT)

    for i in range(ni):
        for j in range(nj):
            k = flow_dir[i, j]
            if k == -1:
                receiver[i, j] = [-1, -1]
            else:
                r_i = i + di[k]
                r_j = j + dj[k]
                receiver[i, j] = [r_i, r_j]

    return receiver


@_speed_up
def _add_to_stack(i, j,
                  receiver: np.ndarray,
                  ordered_nodes: np.ndarray,
                  stack_size,
                  in_list: np.ndarray):
    if i == -1:
        # reach nodata, which is edge.
        # Theoraticall, this should never occur, because outlet's receiver is itself.
        return ordered_nodes, stack_size
    if in_list[i, j]:
        return ordered_nodes, stack_size
    if receiver[i, j, 0] != i or receiver[i, j, 1] != j:
        ordered_nodes, stack_size = _add_to_stack(receiver[i, j, 0], receiver[i, j, 1],
                                                  receiver, ordered_nodes,
                                                  stack_size, in_list)

    ordered_nodes[stack_size, 0] = i
    ordered_nodes[stack_size, 1] = j
    stack_size += 1
    in_list[i, j] = True

    return ordered_nodes, stack_size


@_speed_up
def _is_head_impl(receiver: np.ndarray):
    ni, nj, _ = receiver.shape

    is_head = np.ones((ni, nj))
    for i in range(ni):
        for j in range(nj):
            if receiver[i, j, 0] == -1:  # nodata
                is_head[i, j] = False
            elif receiver[i, j, 0] != i or receiver[i, j, 1] != j:
                is_head[receiver[i, j, 0], receiver[i, j, 1]] = False

    return is_head


@_speed_up
def _build_ordered_array_impl(receiver: np.ndarray):
    """Implementation of build_ordered_array"""
    ni, nj, _ = receiver.shape

    is_head = _is_head_impl(receiver)

    in_list = np.zeros((ni, nj))
    stack_size = 0
    ordered_nodes = np.zeros((ni*nj, 2), dtype=INT)
    for i in range(ni):
        for j in range(nj):
            if is_head[i, j]:
                ordered_nodes, stack_size = _add_to_stack(i, j,
                                                          receiver, ordered_nodes,
                                                          stack_size, in_list)

    ordered_nodes = ordered_nodes[:stack_size]

    # currently ordered_nodes is downstream-to-upstream,
    # we want to reverse it to upstream-to-downstream because it's more intuitive.
    ordered_nodes = ordered_nodes[::-1]
    return ordered_nodes


@_speed_up
def _flow_accumulation_impl(receiver: np.ndarray, ordered_nodes: np.ndarray, cellsize):
    ni, nj, _ = receiver.shape
    drainage_area = np.ones((ni, nj)) * cellsize
    for k in range(len(ordered_nodes)):
        i, j = ordered_nodes[k]
        r_i, r_j = receiver[i, j]
        if r_i == -1:
            # skip because (i, j) is already the outlet
            continue
        if r_i != i or r_j != j:
            drainage_area[r_i, r_j] += drainage_area[i, j]

    return drainage_area


@_speed_up
def _extract_stream_from_receiver_impl(i, j, receiver: np.ndarray):
    if receiver[i, j, 0] == -1:
        raise RuntimeError("Invalid coords i, j. This is a nodata point.")

    # numba does not work with python list, so we allocate a np.ndarray here,
    # then change its size later if necessary
    stream_coords = np.zeros((10000, 2), dtype=INT)

    stream_coords[0] = [i, j]
    k = 1

    r_i, r_j = receiver[i, j]
    while r_i != i or r_j != j:
        if len(stream_coords) > k:
            stream_coords[k] = [r_i, r_j]
            k += 1
        else:
            stream_coords = np.vstack((stream_coords, np.zeros((10000, 2), dtype=INT)))
            stream_coords[k] = [r_i, r_j]
            k += 1
        i, j = r_i, r_j
        r_i, r_j = receiver[i, j]

    stream_coords = stream_coords[:k]
    return stream_coords


@_speed_up
def _build_catchment_mask_impl(outlet_i, outlet_j, receiver: np.ndarray, ordered_nodes: np.ndarray):
    nrow, ncol, _ = receiver.shape
    mask = np.zeros((nrow, ncol), dtype=np.int8)

    mask[outlet_i, outlet_j] = 1
    # This is probably time-expensive because we need to look the whole list.
    # Building a donor array is another choice, but it will take a lot of space.
    for k in range(len(ordered_nodes)-1, -1, -1):
        i, j = ordered_nodes[k]
        if mask[receiver[i, j, 0], receiver[i, j, 1]]:
            mask[i, j] = 1

    return mask
