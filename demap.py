from ast import Str
from dis import dis
from turtle import st
import numpy as np
import rasterio as rio
import richdem
import copy

try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

def _speed_up(func):
    """A conditional decorator that use numba to speed up the function"""
    if USE_NUMBA:
        return numba.njit(func)
    else:
        return func

class GeoArray:
    """A array with georeferencing and metadata"""

    def __init__(self, data, crs, transform, metadata, nodata=None):
        self.data = data
        self.crs = crs
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
    
class Stream:
    """A stream"""
    
    def __init__(self, coords):
        self.coords = coords # n by 2 np.ndarray
        self.dist_up = None # upstream distance
        
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

def load(filename):
    """Read a geospatial data"""

    with rio.open(filename) as dataset:
        data = dataset.read(1) # DEM data only have 1 band.
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
        flow_dir: a GeoArray contain the flow direction information.
            -1 -- nodata
            0 -- this node produces no flow, i.e., local sink
            1-8 -- flow direction coordinates
    
            Flow coordinates follow RichDEM's style:
            |2|3|4|
            |1|0|5|
            |8|7|6|
            
        receiver: an GeoArray that stores receiver node information.
            Each element is a 1 x 2 array that denotes a pair of indices
            in the associated GeoArray.
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
    
    receiver = build_receiver(flow_dir)
    
    return flow_dir, receiver

def build_receiver(flow_dir: GeoArray):
    
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
    ordered_nodes = _build_ordered_array_impl(receiver.data)
    
    return ordered_nodes

def flow_accumulation(receiver: GeoArray, ordered_nodes: np.ndarray):
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

def extract_stream_network(receiver: GeoArray, ordered_nodes: np.ndarray, drainage_area: GeoArray, drainage_area_threshold=1e6):
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
                stream_coords = _extract_stream_impl(i, j, valid_receiver_data)
                stream = Stream(coords=stream_coords)
                stream.get_upstream_distance(dx, dy)
                stream_network.append(stream)
    
    return stream_network

def get_value_along_stream(stream: Stream, grid: GeoArray):
    i_list = stream.coords[:, 0]
    j_list = stream.coords[:, 1]

    return grid.data[i_list, j_list]

def extract_stream(x, y, receiver: GeoArray):
    #TODO
    pass

@_speed_up
def _extract_stream_impl(i, j, receiver: np.ndarray):
    if receiver[i, j, 0] == -1:
        raise RuntimeError("Invalid coords i, j. This is a nodata point.")
    
    # numba does not work with python list, so we allocate a np.ndarray here,
    # then change its size later if necessary
    stream_coords = np.zeros((10000, 2), dtype=np.int16)
    
    stream_coords[0] = [i, j]
    k = 1
    
    r_i, r_j = receiver[i, j]
    while r_i != i or r_j != j:
        if len(stream_coords) > k:
            stream_coords[k] = [r_i, r_j]
            k += 1
        else:
            stream_coords = np.vstack((stream_coords, np.zeros((10000, 2), dtype=np.int16)))
            stream_coords[k] = [r_i, r_j]
            k += 1
        i, j = r_i, r_j
        r_i, r_j = receiver[i, j]
    
    stream_coords = stream_coords[:k]
    return stream_coords

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
def _build_receiver_impl(flow_dir: np.ndarray):
    """Implementation of build_receiver"""
    di = [0, 0, -1, -1, -1, 0, 1, 1, 1]
    dj = [0, -1, -1, 0, 1, 1, 1, 0, -1]
    
    ni, nj = flow_dir.shape
    receiver = np.zeros((ni, nj, 2), dtype=np.int32)
    
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
    
    ordered_nodes[stack_size] = [i, j]
    stack_size += 1
    in_list[i, j] = True
    
    return ordered_nodes, stack_size

@_speed_up
def _is_head_impl(receiver: np.ndarray):
    ni, nj, _ = receiver.shape

    is_head = np.ones((ni, nj))
    for i in range(ni):
        for j in range(nj):
            if receiver[i, j, 0] == -1: # nodata
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
    ordered_nodes = np.zeros((ni*nj, 2), dtype=np.int32)
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