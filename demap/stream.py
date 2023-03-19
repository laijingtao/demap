import numpy as np
import xarray as xr
import copy
from typing import Union

from .helpers import rowcol_to_xy, xy_to_rowcol, latlon_to_xy, xy_to_latlon
from ._base import is_verbose, INT
from .geogrid import GeoGrid
from ._impl import (_build_ordered_array_impl,
                    _build_pseudo_receiver_from_network_impl,
                    _calculate_dist_up_impl,
                    _build_upward_sub_network_mask)


class _StreamBase:
    """Base class for stream and stream network"""

    def __init__(self, rows, cols, **kwargs):
        self.dataset = xr.Dataset(
            coords={
                "flow_order": np.arange(len(rows), dtype=INT),
                "rows": (["flow_order"], rows),
                "cols": (["flow_order"], cols),
            },
        )
        self._build_hashmap()
        self.dataset.attrs['crs'] = kwargs.get('crs', None)
        self.dataset.attrs['transform'] = kwargs.get('transform', None)

    def __str__(self):
        return f'Stream\n{self.dataset.__str__()}'

    def __repr__(self):
        return f'Stream\n{self.dataset.__repr__()}'

    def _ordered_nodes(self):
        return np.array([self.dataset['rows'].data, self.dataset['cols'].data]).transpose()

    def crs(self):
        return self.dataset.attrs['crs']

    def transform(self):
        return self.dataset.attrs['transform']

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.dataset.attrs['transform'])

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.dataset.attrs['transform'])

    def latlon_to_xy(self, lat, lon):
        return latlon_to_xy(lat, lon, self.dataset.attrs['crs'])

    def xy_to_latlon(self, x, y):
        return xy_to_latlon(x, y, self.dataset.attrs['crs'])

    def dx(self):
        return np.abs(self.dataset.attrs['transform'][0])

    def dy(self):
        return np.abs(self.dataset.attrs['transform'][4])

    def _build_hashmap(self, rows=None, cols=None):
        #assert isinstance(self.ordered_nodes, np.ndarray), 'Missing ordered_nodes or wrong type'

        nodes_hash = {}
        if rows is None or cols is None:
            rows = self.dataset['rows'].data
            cols = self.dataset['cols'].data
        for k in range(len(rows)):
            row, col = rows[k], cols[k]
            nodes_hash['{}_{}'.format(int(row), int(col))] = k

        self._nodes_hash = nodes_hash

    def index_of(self, row, col):
        return self._nodes_hash['{}_{}'.format(int(row), int(col))]

    def nearest_to_xy(self, x, y):
        """
        Return the stream node nearest to given x, y geographic coordinates.
        """
        #row, col = self.xy_to_rowcol(x, y)
        #d_i = np.abs(self.dataset['rows'].data - row)
        #d_j = np.abs(self.dataset['cols'].data - col)

        x_list, y_list = self.rowcol_to_xy(self.dataset['rows'].data, self.dataset['cols'].data)
        d_x = np.abs(x_list - x)
        d_y = np.abs(y_list - y)

        dist = np.sqrt(np.power(d_x, 2) + np.power(d_y, 2))
        k = np.argmin(dist)

        return self.dataset['rows'].data[k], self.dataset['cols'].data[k]

    def get_value(self, data_source: Union[GeoGrid, 'StreamNetwork', np.ndarray],
                  var_name=None):
        if not isinstance(data_source, (GeoGrid, StreamNetwork, np.ndarray)):
            raise TypeError("Unsupported data_source type")

        if isinstance(data_source, GeoGrid):
            i_list = self.dataset['rows'].data
            j_list = self.dataset['cols'].data
            val = data_source.dataarray.data[i_list, j_list]

        if isinstance(data_source, np.ndarray):
            i_list = self.dataset['rows'].data
            j_list = self.dataset['cols'].data
            val = data_source[i_list, j_list]

        if isinstance(data_source, StreamNetwork):
            if var_name is None:
                raise ValueError("var_name cannot be None when getting data from StreamNetwork")
            else:
                nrows = data_source.dataset['rows'].data.max()+1
                ncols = data_source.dataset['cols'].data.max()+1
                if var_name in data_source.dataset:
                    source_dataarray = data_source.dataset[var_name]
                elif var_name in data_source.dataset.attrs:
                    source_dataarray = data_source.dataset.attrs[var_name]
                else:
                    raise KeyError('{} not found'.format(var_name))
                val_grid = np.empty((nrows, ncols), dtype=source_dataarray.dtype)
                i_list_source = data_source.dataset['rows'].data
                j_list_source = data_source.dataset['cols'].data
                val_grid[i_list_source, j_list_source] = source_dataarray.data

                i_list = self.dataset['rows'].data
                j_list = self.dataset['cols'].data
                val = val_grid[i_list, j_list]

                '''
                index_of = data_source.index_of
                val = np.zeros(len(self.dataset['flow_order']))
                for k in range(len(self.dataset['flow_order'])):
                    i, j = self.dataset['rows'][k], self.dataset['cols'][k]
                    val[k] = data_source.dataset[var_name].data[index_of(i, j)]
                '''

        if var_name is not None:
            self.dataset[var_name] = (('flow_order'), val)

        return val


class Stream(_StreamBase):
    """A stream"""

    def __init__(self, rows, cols, **kwargs):
        _StreamBase.__init__(self, rows, cols, **kwargs)

        if self.dataset.attrs['transform'] is not None:
            _ = self.get_upstream_distance()
        else:
            print("Warning: no transform info for this stream")

    def get_upstream_distance(self):
        if self.dataset.attrs['transform'] is None:
            raise RuntimeError("No transform info for this stream,\
                cannot calculate the distance")

        if len(self.dataset['flow_order']) == 1:
            dist_up = np.array([0.])
            self.dataset['distance_upstream'] = (["flow_order"], dist_up)
            return self.dataset['distance_upstream']

        dist_up = np.zeros(len(self.dataset['flow_order']))
        dist_down = np.zeros(len(self.dataset['flow_order']))

        i_list = self.dataset['rows'].data
        j_list = self.dataset['cols'].data

        x_list, y_list = self.rowcol_to_xy(i_list, j_list)

        d_x = np.abs(x_list[:-1] - x_list[1:])
        d_y = np.abs(y_list[:-1] - y_list[1:])
        d_dist = np.sqrt(np.power(d_x, 2) + np.power(d_y, 2))

        dist_down[1:] = np.cumsum(d_dist)
        dist_up = dist_down[-1] - dist_down

        self.dataset['distance_upstream'] = (["flow_order"], dist_up)
        return self.dataset['distance_upstream']

    def length(self):
        return self.dataset['distance_upstream'].data[0] - self.dataset['distance_upstream'].data[-1]

    def dir_vector_at_rowcol(self, row, col, smooth_range=1e3):
        idx = self.index_of(row, col)
        dist_up = self.dataset['distance_upstream'].data
        in_range = np.abs(dist_up - dist_up[idx]) <= smooth_range*0.5

        k = idx
        while k > 0 and in_range[k]:
            k -= 1
        row_up, col_up = self.dataset['rows'].data[k], self.dataset['cols'].data[k]

        k = idx
        while k < len(self.dataset['flow_order']) - 1 and in_range[k]:
            k += 1
        row_down, col_down = self.dataset['rows'].data[k], self.dataset['cols'].data[k]

        x_up, y_up = self.rowcol_to_xy(row_up, col_up)
        x_down, y_down = self.rowcol_to_xy(row_down, col_down)

        dir_vector = np.array([x_down - x_up, y_down - y_up])

        return dir_vector

    def mean_value_at_rowcol(self, row, col, value, smooth_range=1e3):
        idx = self.index_of(row, col)
        dist_up = self.dataset['distance_upstream'].data
        in_range = np.abs(dist_up - dist_up[idx]) <= smooth_range*0.5

        '''
        k = idx
        while k > 0 and in_range[k]:
            k -= 1
        idx_up = k

        k = idx
        while k < len(self.dataset['flow_order']) - 1 and in_range[k]:
            k += 1
        idx_down = k

        return np.mean(value[idx_up:idx_down+1])
        '''
        return np.mean(value[in_range == True])

    def smooth_value(self, value, smooth_range=1e3):
        if isinstance(value, str):
            value = self.dataset[value]
        
        smooth_value = value.copy()
        rows = self.dataset['rows'].data
        cols = self.dataset['cols'].data
        for k in range(len(smooth_value)):
            row, col = rows[k], cols[k]
            smooth_value[k] = self.mean_value_at_rowcol(row, col, value, smooth_range)

        return smooth_value

    def smooth_profile(self, dem: Union[GeoGrid, np.ndarray], **kwargs):
        """Return a smoothed channel profile by removing obstacle

        Parameters
        ----------
        dem : Union[GeoGrid, np.ndarray]
            DEM

        Returns
        -------
        z: array-like
            smoothed channel profile
        """
        z = self.get_value(dem)
        z = z.astype(float)

        for k in range(1, len(z)):
            if z[k] > z[k-1]:
                z[k] = np.nextafter(z[k-1], z[k-1]-1)

        return z


class StreamNetwork(_StreamBase):
    """
    A StreamNetwork object saves the stream network using link information

    `flow_order` saves the upstream-to-downstream order of all the nodes in this network.
    
    `rows` and `cols` save the coordinates of node in matrix.

    `downstream` saves the downstream node info.
    `downstream[i]` is the `flow_order` of the node `i` (`i` is also the `flow_order`)

    if `downstream` is -1, then this is an outlet.
    if `
    """

    

    def __init__(self, receiver: GeoGrid = None, **kwargs):
        if receiver is not None:
            rows, cols, downstream, crs, transform = self._build_from_receiver(receiver)
            _StreamBase.__init__(self, rows, cols, crs=crs, transform=transform)
            self.dataset['downstream'] = (['flow_order'], downstream)
        elif all(k in kwargs for k in ('rows', 'cols', 'downstream')):
            rows = np.array(kwargs['rows'], dtype=INT)
            cols = np.array(kwargs['cols'], dtype=INT)
            crs = kwargs.get('crs', None)
            transform = kwargs.get('transform', None)
            _StreamBase.__init__(self, rows, cols, crs=crs, transform=transform)
            self.dataset['downstream'] = (['flow_order'], np.array(kwargs['downstream'], dtype=INT))
        else:
            print("Warning: incomplete input, an empty StreamNetwork was created")
            _StreamBase.__init__(self, rows=[], cols=[])

        if self.dataset.attrs['transform'] is not None:
            _ = self.get_upstream_distance()

    def __str__(self):
        return f'StreamNetwork\n{self.dataset.__str__()}'

    def __repr__(self):
        return f'StreamNetwork\n{self.dataset.__repr__()}'

    def _build_from_receiver(self, receiver: GeoGrid):
        transform = copy.deepcopy(receiver.dataarray.attrs['transform'])
        crs = copy.deepcopy(receiver.dataarray.attrs['crs'])

        ordered_nodes = _build_ordered_array_impl(receiver.dataarray.data)
        rows = ordered_nodes[:, 0]
        cols = ordered_nodes[:, 1]
        n_nodes = len(ordered_nodes)

        self._build_hashmap(rows=rows, cols=cols)

        # receiver_list follow the order of ordered_nodes
        receiver_list = receiver.dataarray.data[rows, cols]

        downstream = -np.ones(n_nodes, dtype=INT)

        index_of = self.index_of
        for k in range(len(receiver_list)):
            from_idx = index_of(rows[k], cols[k])
            to_idx = index_of(receiver_list[k, 0], receiver_list[k, 1])

            if from_idx != to_idx:  # not outlet
                downstream[from_idx] = to_idx

        return rows, cols, downstream, crs, transform

    def get_upstream_distance(self):
        ordered_nodes = self._ordered_nodes()
        downstream = self.dataset['downstream'].data
        pseudo_receiver = _build_pseudo_receiver_from_network_impl(
            ordered_nodes, downstream)

        dist_up_grid = _calculate_dist_up_impl(
            pseudo_receiver, ordered_nodes, self.dx(), self.dy())

        dist_up = self.get_value(dist_up_grid)

        self.dataset['distance_upstream'] = (["flow_order"], dist_up)
        return self.dataset['distance_upstream']

    def to_streams(self, mode='tributary'):
        if is_verbose():
            print("Converting stream network to streams ...")

        if mode not in ['all', 'tributary']:
            raise ValueError("Unknow mode, accepted modes are: \'all\', \'tributary\'")

        if mode == 'tributary':
            mode_flag = 1
        else:
            mode_flag = 2

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

            streams.append(new_stream)

        streams = np.array(streams)
        if mode_flag == 1:
            # sort by length again for tributary mode, note it's the length of tributary stream
            length_list = np.array([st.length() for st in streams])
            sort_idx = np.argsort(length_list)[::-1]
            streams = streams[sort_idx]

        return streams

    def extract_from_xy(self, x, y, direction='up'):
        if is_verbose():
            print("Extracting stream network ...")
        if direction not in ['up', 'down']:
            raise ValueError("Unknown direction, \'up\' or \'down\'")

        row, col = self.nearest_to_xy(x, y)
        if direction == 'up':
            return self._extract_from_rowcol_up(row, col)
        elif direction == 'down':
            return self._extract_from_rowcol_down(row, col)

    def _extract_from_rowcol_down(self, row, col):
        index_of = self.index_of
        rows = self.dataset['rows'].data
        cols = self.dataset['cols'].data
        downstream = self.dataset['downstream'].data
        in_sub_network = np.zeros(len(rows), dtype=bool)

        k = index_of(row, col)
        in_sub_network[k] = True

        # go downstream
        while downstream[k] != -1:
            i, j = rows[downstream[k]], cols[downstream[k]]
            k = index_of(i, j)
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

        sub_network = StreamNetwork(rows=sub_rows, cols=sub_cols,
                                    downstream=sub_downstream,
                                    crs=self.dataset.attrs['crs'],
                                    transform=self.dataset.attrs['transform'])

        return sub_network

    def _extract_from_rowcol_up(self, row, col):
        index_of = self.index_of
        rows = self.dataset['rows'].data
        cols = self.dataset['cols'].data
        downstream = self.dataset['downstream'].data

        outlet = index_of(row, col)
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

        sub_network = StreamNetwork(rows=sub_rows, cols=sub_cols,
                                    downstream=sub_downstream,
                                    crs=self.dataset.attrs['crs'],
                                    transform=self.dataset.attrs['transform'])

        return sub_network

    def smooth_profile(self, dem: Union[GeoGrid, np.ndarray], **kwargs):
        """Return a smoothed channel profile by removing obstacle

        Parameters
        ----------
        dem : Union[GeoGrid, np.ndarray]
            DEM

        Returns
        -------
        z: array-like
            smoothed channel profile
        """
        downstream = self.dataset['downstream'].data

        z = self.get_value(dem)
        z = z.astype(float)

        for k in range(1, len(z)):
            if downstream[k] != -1:
                d_k = downstream[k]
                if z[d_k] > z[k]:
                    z[d_k] = np.nextafter(z[k], z[k]-1)

        return z

    def get_stream_order(self, method='strahler'):
        downstream = self.dataset['downstream'].data

        num_of_donor = np.zeros_like(downstream)
        stream_order = np.ones_like(downstream)

        if method == 'strahler':
            for k in range(len(stream_order)):
                if downstream[k] != -1:
                    if num_of_donor[downstream[k]] > 0:
                        if stream_order[k] > stream_order[downstream[k]]:
                            stream_order[downstream[k]] = stream_order[k]
                        elif stream_order[k] == stream_order[downstream[k]]:
                            stream_order[downstream[k]] += 1
                    else:
                        stream_order[downstream[k]] = stream_order[k]
                    num_of_donor[downstream[k]] += 1
        elif method == 'shreve':
            for k in range(len(stream_order)):
                if downstream[k] != -1:
                    if num_of_donor[downstream[k]] > 0:
                        stream_order[downstream[k]] += stream_order[k]
                    else:
                        stream_order[downstream[k]] = stream_order[k]
                    num_of_donor[downstream[k]] += 1
        else:
            raise KeyError('Unkown method \'{}\'. strahler or shreve?'.format(method))

        self.dataset['stream_order'] = (('flow_order'), stream_order)
        return self.dataset['stream_order']

# TODO: this is not compatible with new StreamNetwork, fix later
def merge_stream_network(network1: StreamNetwork, network2: StreamNetwork):

    ordered_nodes = np.append(network1.ordered_nodes, network2.ordered_nodes, axis=0)

    downstream = np.append(network1.downstream, network2.downstream)
    upstream = np.append(network1.upstream, network2.upstream, axis=0)

    n1 = len(network1.ordered_nodes)

    downstream[n1:][downstream[n1:] >= 0] += n1
    upstream[n1:, 1:][upstream[n1:, 1:] >= 0] += n1

    merged = StreamNetwork(ordered_nodes=ordered_nodes,
                           downstream=downstream,
                           upstream=upstream,
                           crs=copy.deepcopy(network1.crs),
                           transform=copy.deepcopy(network1.transform))

    for key in merged.attrs:
        merged.attrs[key] = np.append(network1.attrs[key], network2.attrs[key], axis=0)

    return merged

