import numpy as np
import xarray as xr
import rioxarray
from typing import Union

from .helpers import rowcol_to_xy, xy_to_rowcol, latlon_to_xy, xy_to_latlon
from ._base import _speed_up

class _XarrayAccessorBase:

    def __init__(self, xrobj):
        self._xrobj = xrobj
        self._crs = self._xrobj.rio.crs
        self._transform = self._xrobj.rio.transform()
    
    @property
    def crs(self):
        return self._crs
    
    @crs.setter
    def crs(self, value):
        self._crs = value

    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, value):
        self._transform = value

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def latlon_to_xy(self, lat, lon):
        return latlon_to_xy(lat, lon, self.crs)

    def xy_to_latlon(self, x, y):
        return xy_to_latlon(x, y, self.crs)

    '''
    @property
    def xy(self):
        return np.asarray(self._xrobj['x']), np.asarray(self._xrobj['y'])
    
    
    @property
    def rowcol(self):
        rows, cols = self.xy_to_rowcol(np.asarray(self._xrobj['x']), np.asarray(self._xrobj['y']))
        return rows, cols
    
    @property
    def latlon(self):
        lat, lon = self.xy_to_latlon(np.asarray(self._xrobj['x']), np.asarray(self._xrobj['y']))
        return lat, lon
    '''

    @property
    def dx(self):
        return np.abs(self.transform[0])

    @property
    def dy(self):
        return np.abs(self.transform[4])


@xr.register_dataarray_accessor("demap")
class DemapDataarrayAccessor(_XarrayAccessorBase):

    def __init__(self, xrobj):
        super().__init__(xrobj)
        self._nodata = self._xrobj.rio.nodata

    @property
    def nodata(self):
        return self._nodata
    
    @nodata.setter
    def nodata(self, value):
        self._nodata = value


@xr.register_dataset_accessor("demap")
class DemapDatasetAccessor(_XarrayAccessorBase):
    
    def _get_var_data_for_func(self, var_name, local_dict):

        if hasattr(var_name, '__len__') and (not isinstance(var_name, str)):
            var_name_list = var_name
        else:
            var_name_list = [var_name]

        var_data_list = []
        for var in var_name_list:
            if var in self._xrobj:
                var_data = self._xrobj[var].to_numpy()
            elif local_dict[var] is not None:
                var_data = np.asarray(local_dict[var])
            else:
                raise ValueError("{} missing".format(var))
            
            var_data_list.append(var_data)
    
        if len(var_name_list) == 1:
            var_data_list = var_data_list[0]

        return var_data_list


    def get_flow_direction(self):

        dem = self._xrobj['dem']

        dem_cond = _fill_depression(np.asarray(dem), dem.rio.nodata, self.transform)
        flow_dir_data = _flow_dir_from_richdem(dem_cond, dem.rio.nodata, self.transform)

        self._xrobj['dem_cond'] = (('y', 'x'), dem_cond)
        self._xrobj['flow_dir'] = (('y', 'x'), flow_dir_data)

        return self._xrobj['flow_dir']

    
    def build_hydro_order(self, flow_dir: Union[np.ndarray, xr.DataArray] = None):
        """Build a hydrologically ordered list of nodes.
        
        Parameters
        ----------
        flow_dir : numpy array or xarray Dataarray, optional
            Flow direction. By default None.
            If the dataset contains 'flow_dir', the one in dataset will be used.

        Return:
            ordered_nodes: in this array, upstream point is always in front of
                its downstream point. Each element is a 1 x 2 array that
                denotes the (row, col) coordinates in the Dataset.

        """

        flow_dir_data = self._get_var_data_for_func('flow_dir', locals())

        import sys
        ni, nj = flow_dir_data.shape
        sys.setrecursionlimit(ni*nj)

        ordered_nodes = _build_hydro_order_impl(flow_dir_data)

        self._xrobj['ordered_nodes'] = (('hydro_order', 'node_coords'), ordered_nodes)

        return self._xrobj['ordered_nodes']
    
    
    def accumulate_flow(self,
                        flow_dir: Union[np.ndarray, xr.DataArray] = None,
                        ordered_nodes: Union[np.ndarray, xr.DataArray] = None):
        """Flow accumulation
        """

        flow_dir_data, ordered_nodes_data = self._get_var_data_for_func(['flow_dir', 'ordered_nodes'], locals())
        
        cellsize = self.dx * self.dy
        drainage_area_data = _accumulate_flow_impl(flow_dir_data, ordered_nodes_data, cellsize)

        self._xrobj['drainage_area'] = (('y', 'x'), drainage_area_data)

        return self._xrobj['drainage_area']
    

    #####################################
    # stream methods below
    #####################################

    @property
    def stream_coords_rowcol(self):
        return np.asarray(self._xrobj['rows']), np.asarray(self._xrobj['cols'])
    
    @property
    def stream_coords_xy(self):
        rows, cols = np.asarray(self._xrobj['rows']), np.asarray(self._xrobj['cols'])
        x, y = self.rowcol_to_xy(rows, cols)
        return x, y
    
    @property
    def stream_coords_latlon(self):
        rows, cols = np.asarray(self._xrobj['rows']), np.asarray(self._xrobj['cols'])
        x, y = self.rowcol_to_xy(rows, cols)
        lat, lon = self.xy_to_latlon(x, y)
        return lat, lon

    def _build_index_hash(self):
        index_hash = {}

        rows, cols = self.stream_coords_rowcol

        for k in range(len(rows)):
            row, col = rows[k], cols[k]
            index_hash['{}_{}'.format(int(row), int(col))] = k

        self._index_hash = index_hash

        return self._index_hash
    
    def _index_in_ordered_array(self, row, col):

        if not hasattr(row, '__len__'):
            row = np.asarray([row])
            col = np.asarray([col])

        try:
            index_hash = getattr(self, '_index_hash')
            if index_hash is None:
                index_hash = self._build_index_hash()
        except AttributeError:
            index_hash = self._build_index_hash()
        
        #index_list = _index_in_ordered_array_impl(row, col, ordered_nodes)
        index_list = [index_hash['{}_{}'.format(int(row[k]), int(col[k]))] for k in range(len(row))]

        if len(index_list) == 1:
            index_list = index_list[0]
        
        return index_list


    def build_stream_network(self,
                             flow_dir: Union[np.ndarray, xr.DataArray] = None,
                             drainage_area: Union[np.ndarray, xr.DataArray] = None,
                             drainage_area_threshold=1e6):

        flow_dir_data, drainage_area_data = self._get_var_data_for_func(['flow_dir', 'drainage_area'], locals())

        # all nodes with drainage_area smaller than the threshold are set as nodata
        flow_dir_data = np.where(drainage_area_data > drainage_area_threshold, flow_dir_data, -1)

        #stream_network = self._build_stream_network_impl(flow_dir_data)

        ordered_nodes = _build_hydro_order_impl(flow_dir_data)

        n_nodes = len(ordered_nodes)

        rows = ordered_nodes[:, 0]
        cols = ordered_nodes[:, 1]

        flow_dir_list = flow_dir_data[rows, cols]
        flow_dir_list = np.where(flow_dir_list >= 0, flow_dir_list, 0) # just to make sure there is no negative in flow_dir_list

        """
        Demap's flow direction coding:
        |4|3|2|
        |5|0|1|
        |6|7|8|
        """
        di = np.array([0, 0, -1, -1, -1, 0, 1, 1, 1])
        dj = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])

        rcv_rows = rows + di[flow_dir_list]
        rcv_cols = cols + dj[flow_dir_list]

        # fix if rcv_row, rcv_col is nodata
        rcv_rows = np.where(flow_dir_data[rcv_rows, rcv_cols] >= 0, rcv_rows, rows)
        rcv_cols = np.where(flow_dir_data[rcv_rows, rcv_cols] >= 0, rcv_cols, cols)

        index_hash = {}
        for k in range(n_nodes):
            row, col = rows[k], cols[k]
            index_hash['{}_{}'.format(int(row), int(col))] = k

        index_list = np.arange(n_nodes)
        rcv_index_list = np.array([index_hash['{}_{}'.format(int(rcv_rows[k]), int(rcv_cols[k]))] for k in range(n_nodes)])
        
        downstream = -np.ones(n_nodes, dtype=np.int32)
        downstream[index_list] = np.where(index_list != rcv_index_list, rcv_index_list, -1)

        stream_x, stream_y = self.rowcol_to_xy(rows, cols)
        distance_upstream = _calculate_dist_up_impl(stream_x, stream_y, downstream)

        #ordered_nodes, downstream = _split_stream_network(ordered_nodes, downstream, distance_upstream)


        stream_network_ds = xr.Dataset(
            coords={
                "hydro_order": np.arange(n_nodes, dtype=np.int32),
            },
        )
        stream_network_ds["rows"] = (["hydro_order"], rows)
        stream_network_ds["cols"] = (["hydro_order"], cols)
        stream_network_ds['downstream'] = (['hydro_order'], downstream)
        stream_network_ds['distance_upstream'] = (['hydro_order'], distance_upstream)
        stream_network_ds.attrs['crs'] = self.crs
        stream_network_ds.attrs['transform'] = self.transform

        return stream_network_ds
    
    
    def split_stream_network(self, mode='tributary'):
        if mode not in ['all', 'tributary']:
            raise ValueError("Unknow mode, accepted modes are: \'all\', \'tributary\'")

        if mode == 'tributary':
            mode_flag = 1
        else:
            mode_flag = 2

        rows, cols = self.stream_coords_rowcol
        ordered_nodes = np.vstack([rows, cols]).T
        downstream = np.asarray(self._xrobj['downstream'])
        distance_upstream = np.asarray(self._xrobj['distance_upstream'])

        splitted_stream_idx = _split_stream_network(ordered_nodes, downstream, distance_upstream, mode_flag)

        stream_end_idx = np.where(splitted_stream_idx == -1)[0]
        streams = []
        for k in range(len(stream_end_idx)):
            if k == 0:
                stream_start = 0
            else:
                stream_start = stream_end_idx[k-1]+1

            stream_end = stream_end_idx[k]

            stream_idx = splitted_stream_idx[stream_start:stream_end]
            sub_ordered_nodes = ordered_nodes[stream_idx]
            sub_distance_upstream = distance_upstream[stream_idx]
            
            stream_ds = xr.Dataset(
                coords={
                    "hydro_order": np.arange(len(sub_ordered_nodes), dtype=np.int32),
                },
            )
            stream_ds["rows"] = (["hydro_order"], sub_ordered_nodes[:, 0])
            stream_ds["cols"] = (["hydro_order"], sub_ordered_nodes[:, 1])
            # TODO: maybe also add downstream for streams
            stream_ds['distance_upstream'] = (['hydro_order'], sub_distance_upstream)
            stream_ds.attrs['crs'] = self.crs
            stream_ds.attrs['transform'] = self.transform

            streams.append(stream_ds)

        #streams = np.array(streams)
        
        #if mode_flag == 1:
        # sort by length again for tributary mode, note it's the length of tributary stream
        length_list = np.array([np.max(st['distance_upstream'])-np.min(st['distance_upstream']) for st in streams])
        sort_idx = np.argsort(length_list)[::-1]
        streams = [streams[i] for i in sort_idx]
        
        return streams
    
    def get_value(self, data_source: Union[xr.DataArray, xr.Dataset, np.ndarray],
                  var_name=None):
        if not isinstance(data_source, (xr.DataArray, xr.Dataset, np.ndarray)):
            raise TypeError("Unsupported data_source type")

        # grid data
        if isinstance(data_source, (xr.DataArray, np.ndarray)):
            i_list, j_list = self.stream_coords_rowcol
            val = np.asarray(np.asarray(data_source)[i_list, j_list])

        # stream data
        if isinstance(data_source, xr.Dataset):
            if var_name is None:
                raise ValueError("var_name cannot be None when getting data from streams")
            else:
                source_dataarray = data_source[var_name]
                i_list, j_list = self.stream_coords_rowcol
                index_list = data_source.demap._index_in_ordered_array(i_list, j_list)
                val = source_dataarray.data[index_list]

        if var_name is not None:
            self._xrobj[var_name] = (('hydro_order'), val)

        return val

    def nearest_to_xy(self, x, y):
        """
        Return the stream node nearest to given x, y geographic coordinates.
        """
        #row, col = self.xy_to_rowcol(x, y)
        #d_i = np.abs(self.dataset['rows'].data - row)
        #d_j = np.abs(self.dataset['cols'].data - col)

        x_list, y_list = self.rowcol_to_xy(np.asarray(self._xrobj['rows']), np.asarray(self._xrobj['cols']))
        d_x = np.abs(x_list - x)
        d_y = np.abs(y_list - y)

        dist = np.sqrt(np.power(d_x, 2) + np.power(d_y, 2))
        k = np.argmin(dist)

        return k


    def extract_from_xy(self, x, y, direction='up'):
        if direction not in ['up', 'down']:
            raise ValueError("Unknown direction, \'up\' or \'down\'")

        node_idx = self.nearest_to_xy(x, y)
        if direction == 'up':
            return self._extract_from_rowcol_up(node_idx)
        elif direction == 'down':
            return self._extract_from_rowcol_down(node_idx)


    def _extract_from_rowcol_up(self, node_idx):
        import copy

        rows = np.asarray(self._xrobj['rows'])
        cols = np.asarray(self._xrobj['cols'])
        downstream = np.asarray(self._xrobj['downstream'])

        outlet = node_idx
        in_sub_network = _build_upward_sub_network_mask(outlet, downstream)

        sub_rows = copy.deepcopy(rows[in_sub_network == True])
        sub_cols = copy.deepcopy(cols[in_sub_network == True])
        sub_downstream = copy.deepcopy(downstream[in_sub_network == True])

        # update index in sub_downstream
        new_idx = -np.ones(len(rows))
        new_idx[in_sub_network == True] = np.arange(len(sub_rows))
        # new_idx is essentially a function that convert old index to new index
        sub_downstream[np.where(sub_downstream >= 0)] =\
            new_idx[sub_downstream[np.where(sub_downstream >= 0)]]
        
        sub_network_ds = xr.Dataset(
                coords={
                    "hydro_order": np.arange(len(sub_rows), dtype=np.int32),
                },
            )
        sub_network_ds["rows"] = (["hydro_order"], sub_rows)
        sub_network_ds["cols"] = (["hydro_order"], sub_cols)
        sub_network_ds['downstream'] = (['hydro_order'], sub_downstream)
        sub_network_ds['distance_upstream'] = (['hydro_order'], np.asarray(self._xrobj['distance_upstream'])[in_sub_network == True])
        sub_network_ds.attrs['crs'] = self.crs
        sub_network_ds.attrs['transform'] = self.transform

        return sub_network_ds
    
    
    def _extract_from_rowcol_down(self, node_idx):
        import copy

        rows = np.asarray(self._xrobj['rows'])
        cols = np.asarray(self._xrobj['cols'])
        downstream = np.asarray(self._xrobj['downstream'])
        in_sub_network = np.zeros(len(rows), dtype=bool)

        k = node_idx
        in_sub_network[k] = True

        # go downstream
        while downstream[k] != -1:
            k = downstream[k]
            in_sub_network[k] = True

        sub_rows = copy.deepcopy(rows[in_sub_network == True])
        sub_cols = copy.deepcopy(cols[in_sub_network == True])
        sub_downstream = copy.deepcopy(downstream[in_sub_network == True])

        # update index in sub_downstream
        new_idx = -np.ones(len(rows))
        new_idx[in_sub_network == True] = np.arange(len(sub_rows))
        # new_idx is essentially a function that convert old index to new index
        sub_downstream[np.where(sub_downstream >= 0)] =\
            new_idx[sub_downstream[np.where(sub_downstream >= 0)]]
        
        sub_network_ds = xr.Dataset(
                coords={
                    "hydro_order": np.arange(len(sub_rows), dtype=np.int32),
                },
            )
        sub_network_ds["rows"] = (["hydro_order"], sub_rows)
        sub_network_ds["cols"] = (["hydro_order"], sub_cols)
        sub_network_ds['downstream'] = (['hydro_order'], sub_downstream)
        sub_network_ds['distance_upstream'] = (['hydro_order'], np.asarray(self._xrobj['distance_upstream'])[in_sub_network == True])
        sub_network_ds.attrs['crs'] = self.crs
        sub_network_ds.attrs['transform'] = self.transform

        return sub_network_ds
    
    
@_speed_up
def _index_in_ordered_array_impl(row, col, ordered_nodes):

    index_list = np.zeros(len(row), dtype=np.int32)
    for i in range(len(index_list)):
        index_list[i] = np.where(np.logical_and(ordered_nodes[:, 0] == row[i], ordered_nodes[:, 1] == col[i]))[0].astype(np.int32)[0]

    return index_list


@_speed_up
def _calculate_dist_up_impl(x: np.ndarray, y: np.ndarray, downstream: np.ndarray):
    dist_up = np.zeros_like(x)

    for k in range(len(x)-1, -1, -1):
        if downstream[k] > -1:
            d_k = downstream[k]
            d_dist = np.sqrt(np.power(x[k] - x[d_k], 2) + np.power(y[k] - y[d_k], 2))
            dist_up[k] = dist_up[d_k] + d_dist

    return dist_up

@_speed_up
def _split_stream_network(ordered_nodes, downstream, distance_upstream, mode_flag):
    n_nodes = len(ordered_nodes)
    is_head = np.ones(n_nodes, dtype=np.bool_)
    for k in range(n_nodes):
        if downstream[k] > -1:
            is_head[downstream[k]] = False
    head_nodes_idx = np.where(is_head == True)[0]

    sort_idx = np.argsort(distance_upstream[head_nodes_idx])[::-1]
    head_nodes_idx = head_nodes_idx[sort_idx]

    if mode_flag == 1:
        new_size = n_nodes*2
    else:
        new_size = len(head_nodes_idx) * n_nodes
    
    splitted_stream_idx = np.zeros(new_size, dtype=np.int32)

    in_streams = np.zeros(n_nodes, dtype=np.bool_)
    length = -1
    for head in head_nodes_idx:
        k = head

        length = length + 1
        splitted_stream_idx[length] = k
        
        while (downstream[k] != -1) and (not in_streams[downstream[k]]):
            k = downstream[k]
            
            length = length + 1
            splitted_stream_idx[length] = k
            
            if mode_flag == 1:
                in_streams[k] = True

        if mode_flag == 1 and downstream[k] != -1:
            # also add juction node
            length = length + 1
            splitted_stream_idx[length] = downstream[k]
        
        length = length + 1 
        splitted_stream_idx[length] = -1 # splitter

    splitted_stream_idx = splitted_stream_idx[:length+1]

    return splitted_stream_idx


def _to_rdarray(grid_data: np.ndarray, nodata, transform):
    import richdem

    if np.sum(np.isnan(grid_data)) > 0:
        raise ValueError("Invalid input for richdem: the grid contains nan value")
    
    if np.isnan(nodata):
        raise ValueError("Invalid nodata value for richdem: {}".format(nodata))
    
    out_rd = richdem.rdarray(grid_data, no_data=nodata)
    out_rd.geotransform = transform.to_gdal()

    return out_rd


def _fill_depression(dem: np.ndarray, nodata, transform):
    """Fill dipression using richDEM

    There are two ways to do this (see below) and Barnes2014 is the default.

    Epsilon filling will give a non-flat filled area:
    https://richdem.readthedocs.io/en/latest/depression_filling.html#epsilon-filling

    Barnes2014 also gives a not-flat filled area,
    and it produces more pleasing view of stream network
    https://richdem.readthedocs.io/en/latest/flat_resolution.html#
    """
    import richdem

    dem_rd = _to_rdarray(dem, nodata, transform)
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

    return dem_rd_filled


def _flow_dir_from_richdem(dem: np.ndarray, nodata, transform):
    """
    Return:
        flow_dir: a grid contain the flow direction information.
        -1 -- nodata
        0 -- this node produces no flow, i.e., local sink
        1-8 -- flow direction coordinates

        RichDEM's coding:
        |2|3|4|
        |1|0|5|
        |8|7|6|

        Demap's flow direction coding:
        |4|3|2|
        |5|0|1|
        |6|7|8|
    """
    import richdem

    dem_rd = _to_rdarray(dem, nodata, transform)

    nodata_flow_dir = -1

    print("RichDEM flow direction output:")

    flow_prop = richdem.FlowProportions(dem=dem_rd, method='D8')

    flow_prop = np.array(flow_prop)
    node_info = flow_prop[:, :, 0]
    flow_dir_code_rd = np.argmax(flow_prop[:, :, 1:9], axis=2) + 1

    to_demap_coding = np.array([0, 5, 4, 3, 2, 1, 8, 7, 6])

    flow_dir_data = np.where(node_info == 0, to_demap_coding[flow_dir_code_rd], 0)
    flow_dir_data = np.where(node_info == -1, 0, flow_dir_data)
    flow_dir_data = np.where(node_info == -2, nodata_flow_dir, flow_dir_data)

    flow_dir_data = flow_dir_data.astype(np.int8)

    return flow_dir_data


@_speed_up
def _accumulate_flow_impl(flow_dir: np.ndarray, ordered_nodes: np.ndarray, cellsize):
    di = [0, 0, -1, -1, -1, 0, 1, 1, 1]
    dj = [0, 1, 1, 0, -1, -1, -1, 0, 1]

    ni, nj = flow_dir.shape
    drainage_area = np.ones((ni, nj)) * cellsize
    for k in range(len(ordered_nodes)):
        i, j = ordered_nodes[k]
        
        if flow_dir[i, j] == 0:
            # sink, skip
            continue

        r_i = i + di[flow_dir[i, j]]
        r_j = j + dj[flow_dir[i, j]]
        if r_i >= 0 and r_i < ni and r_j >= 0 and r_j < nj:
            drainage_area[r_i, r_j] += drainage_area[i, j]
        else:
            # receiver is out the bound, skip
            continue

    return drainage_area


@_speed_up
def _is_head(flow_dir: np.ndarray):
    """
    Demap's flow direction coding:
    |4|3|2|
    |5|0|1|
    |6|7|8|
    """
    di = [0, 0, -1, -1, -1, 0, 1, 1, 1]
    dj = [0, 1, 1, 0, -1, -1, -1, 0, 1]

    ni, nj = flow_dir.shape

    is_head = np.ones((ni, nj), dtype=np.bool_)
    for i in range(ni):
        for j in range(nj):
            k = flow_dir[i, j]
            if k == -1:  # nodata
                is_head[i, j] = False
            elif k != 0:
                r_i = i + di[k]
                r_j = j + dj[k]
                if r_i >= 0 and r_i < ni and r_j >= 0 and r_j < nj:
                    is_head[r_i, r_j] = False

    return is_head


@_speed_up
def _add_to_stack(i, j,
                  flow_dir: np.ndarray,
                  ordered_nodes: np.ndarray,
                  stack_size,
                  in_list: np.ndarray):
    
    """
    Demap's flow direction coding:
    |4|3|2|
    |5|0|1|
    |6|7|8|
    """
    if flow_dir[i, j] == -1:
        # reach nodata
        # Theoraticall, this should never occur, because nodata are excluded
        # from the whole network.
        return ordered_nodes, stack_size
    
    if in_list[i, j]:
        return ordered_nodes, stack_size

    in_list[i, j] = True

    if flow_dir[i, j] != 0:
        di = [0, 0, -1, -1, -1, 0, 1, 1, 1]
        dj = [0, 1, 1, 0, -1, -1, -1, 0, 1]

        ni, nj = flow_dir.shape

        r_i = i + di[flow_dir[i, j]]
        r_j = j + dj[flow_dir[i, j]]
        if r_i >= 0 and r_i < ni and r_j >= 0 and r_j < nj:
            ordered_nodes, stack_size = _add_to_stack(r_i, r_j,
                                                    flow_dir, ordered_nodes,
                                                    stack_size, in_list)

    ordered_nodes[stack_size, 0] = i
    ordered_nodes[stack_size, 1] = j
    stack_size += 1
    

    return ordered_nodes, stack_size


@_speed_up
def _build_hydro_order_impl(flow_dir: np.ndarray):
    """Implementation of build_hydro_order
    
    Demap's flow direction coding:
    |4|3|2|
    |5|0|1|
    |6|7|8|

    """

    ni, nj = flow_dir.shape

    is_head = _is_head(flow_dir)

    in_list = np.zeros((ni, nj), dtype=np.bool_)
    stack_size = 0
    ordered_nodes = np.zeros((ni*nj, 2), dtype=np.int32)
    for i in range(ni):
        for j in range(nj):
            if is_head[i, j]:
                ordered_nodes, stack_size = _add_to_stack(i, j,
                                                          flow_dir, ordered_nodes,
                                                          stack_size, in_list)

    ordered_nodes = ordered_nodes[:stack_size]

    # currently ordered_nodes is downstream-to-upstream,
    # we want to reverse it to upstream-to-downstream because it's more intuitive.
    ordered_nodes = ordered_nodes[::-1]
    return ordered_nodes


@_speed_up
def _build_upward_sub_network_mask(outlet, downstream: np.ndarray):
    n_nodes = len(downstream)
    in_sub_network = np.zeros(n_nodes, dtype=np.bool_)
    in_sub_network[outlet] = True

    # find all channel heads
    is_head = np.ones(n_nodes, dtype=np.bool_)
    for k in range(n_nodes):
        if downstream[k] > -1:
            is_head[downstream[k]] = False

    # check if a chnnel head drains to the target outlet
    for k in range(n_nodes):
        if is_head[k]:
            curr_idx = k
            while curr_idx < outlet and curr_idx > -1:
                curr_idx = downstream[curr_idx]
            if curr_idx == outlet:
                in_sub_network[k] = True

    # now we have all channel heads that drains to the target outlet,
    # add all downstream nodes
    for k in range(n_nodes):
        if is_head[k] and in_sub_network[k]:
            curr_idx = k
            while curr_idx < outlet:
                in_sub_network[curr_idx] = True
                curr_idx = downstream[curr_idx]

    return in_sub_network
