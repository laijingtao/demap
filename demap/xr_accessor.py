import numpy as np
import xarray as xr
import rioxarray
from typing import Union

from .helpers import rowcol_to_xy, xy_to_rowcol, latlon_to_xy, xy_to_latlon
from ._impl import _build_hydro_order_impl, _accumulate_flow_impl
from ._base import _speed_up

@xr.register_dataset_accessor("demap")
class DemapAccessor:
    _ds: xr.Dataset

    def __init__(self, ds):
        self._ds = ds
        self._crs = self._ds.rio.crs
        self._transform = self._ds.rio.transform()
    
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

    @property
    def xy(self):
        return np.asarray(self._ds['x']), np.asarray(self._ds['y'])
    
    '''
    @property
    def rowcol(self):
        rows, cols = self.xy_to_rowcol(np.asarray(self._ds['x']), np.asarray(self._ds['y']))
        return rows, cols
    
    @property
    def latlon(self):
        lat, lon = self.xy_to_latlon(np.asarray(self._ds['x']), np.asarray(self._ds['y']))
        return lat, lon
    '''

    @property
    def dx(self):
        return np.abs(self.transform[0])

    @property
    def dy(self):
        return np.abs(self.transform[4])
    
    def _get_var_data_for_func(self, var_name, local_dict):

        if hasattr(var_name, '__len__') and (not isinstance(var_name, str)):
            var_name_list = var_name
        else:
            var_name_list = [var_name]

        var_data_list = []
        for var in var_name_list:
            if var in self._ds:
                var_data = self._ds[var].to_numpy()
            elif local_dict[var] is not None:
                var_data = np.asarray(local_dict[var])
            else:
                raise ValueError("{} missing".format(var))
            
            var_data_list.append(var_data)
    
        if len(var_name_list) == 1:
            var_data_list = var_data_list[0]

        return var_data_list


    def get_flow_direction(self):

        dem = self._ds['dem']

        flow_dir_data = _flow_dir_from_richdem(np.asarray(dem), dem.rio.nodata, self.transform)

        self._ds['flow_dir'] = (('y', 'x'), flow_dir_data)

        return self._ds['flow_dir']

    
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
        ordered_nodes = _build_hydro_order_impl(flow_dir_data)

        self._ds['ordered_nodes'] = (('hydro_order', 'node_coords'), ordered_nodes)

        return self._ds['ordered_nodes']
    
    
    def accumulate_flow(self,
                        flow_dir: Union[np.ndarray, xr.DataArray] = None,
                        ordered_nodes: Union[np.ndarray, xr.DataArray] = None):
        """Flow accumulation
        """

        flow_dir_data, ordered_nodes_data = self._get_var_data_for_func(['flow_dir', 'ordered_nodes'], locals())
        
        cellsize = self.dx * self.dy
        drainage_area_data = _accumulate_flow_impl(flow_dir_data, ordered_nodes_data, cellsize)

        self._ds['drainage_area'] = (('y', 'x'), drainage_area_data)

        return self._ds['drainage_area']
    
    def _index_in_ordered_array(self, row, col, ordered_nodes):

        if not hasattr(row, '__len__'):
            row = np.asarray([row])
            col = np.asarray([col])
        
        index_list = _index_in_ordered_array_impl(row, col, ordered_nodes)

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

        index_list = self._index_in_ordered_array(rows, cols, ordered_nodes)
        rcv_index_list = self._index_in_ordered_array(rcv_rows, rcv_cols, ordered_nodes)
        
        downstream = -np.ones(len(ordered_nodes), dtype=np.int32)
        downstream[index_list] = np.where(index_list != rcv_index_list, rcv_index_list, -1)


        stream_network_ds = xr.Dataset(
            coords={
                "hydro_order": np.arange(len(rows), dtype=np.int32),
            },
        )
        stream_network_ds["rows"] = (["hydro_order"], rows)
        stream_network_ds["cols"] = (["hydro_order"], cols)
        stream_network_ds['downstream'] = (['hydro_order'], downstream)
        stream_network_ds.attrs['crs'] = self.crs
        stream_network_ds.attrs['transform'] = self.transform

        return stream_network_ds
    
    '''
    def split_stream_network(self, ):
        if mode not in ['all', 'tributary']:
            raise ValueError("Unknow mode, accepted modes are: \'all\', \'tributary\'")

        if mode == 'tributary':
            mode_flag = 1
        else:
            mode_flag = 2

        if var_to_copy is not None:
            if isinstance(var_to_copy, str):
                var_to_copy = [var_to_copy]

        rows = self.dataset['rows'].data
        cols = self.dataset['cols'].data
        n_nodes = len(rows)
        downstream = self.dataset['downstream'].data
        in_streams = np.zeros(n_nodes, dtype=bool)

        is_head = np.ones(n_nodes, dtype=bool)
        for k in range(n_nodes):
            if downstream[k] > -1:
                is_head[downstream[k]] = False
        head_nodes_idx = np.arange(n_nodes)[is_head == True]
        sort_idx = np.argsort(self.dataset['distance_upstream'].data[head_nodes_idx])[::-1]
        head_nodes_idx = head_nodes_idx[sort_idx]

        streams = []
        for head in head_nodes_idx:
            stream_idx = [head]

            # build streams_idx
            k = head
            while (downstream[k] != -1) and (not in_streams[downstream[k]]):
                k = downstream[k]
                stream_idx.append(k)
                if mode_flag == 1:
                    in_streams[k] = True

            if mode_flag == 1 and downstream[k] != -1:
                # also add juction node
                stream_idx.append(downstream[k])

            new_stream = Stream(rows=rows[stream_idx], cols=cols[stream_idx],
                                crs=self.dataset.attrs['crs'],
                                transform=self.dataset.attrs['transform'])

            # make sure the dist_up is relative to outlet of the stream network
            new_stream.dataset['distance_upstream'] += self.dataset['distance_upstream'][stream_idx[-1]]

            if var_to_copy is not None:
                for var in var_to_copy:
                    val = self.dataset[var].data[stream_idx]
                    new_stream.dataset[var] = (('flow_order'), val)

            streams.append(new_stream)

        streams = np.array(streams)
        if mode_flag == 1:
            # sort by length again for tributary mode, note it's the length of tributary stream
            length_list = np.array([st.length() for st in streams])
            sort_idx = np.argsort(length_list)[::-1]
            streams = streams[sort_idx]

        return streams
    '''
    
@_speed_up
def _index_in_ordered_array_impl(row, col, ordered_nodes):

    index_list = np.zeros(len(row), dtype=np.int32)
    for i in range(len(index_list)):
        index_list[i] = np.where(np.logical_and(ordered_nodes[:, 0] == row[i], ordered_nodes[:, 1] == col[i]))[0].astype(np.int32)[0]

    return index_list


def _to_rdarray(grid_data: np.ndarray, nodata, transform):
    import richdem

    if np.sum(np.isnan(grid_data)) > 0:
        raise ValueError("Invalid input for richdem: the grid contains nan value")
    
    if np.isnan(nodata):
        raise ValueError("Invalid nodata value for richdem: {}".format(nodata))
    
    out_rd = richdem.rdarray(grid_data, no_data=nodata)
    out_rd.geotransform = transform.to_gdal()

    return out_rd


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