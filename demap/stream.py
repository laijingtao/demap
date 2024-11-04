import numpy as np
import xarray as xr
from typing import Union

from ._base import _speed_up, _XarrayAccessorBase
from .helpers import round_rowcol_coord

class StreamAccessor(_XarrayAccessorBase):

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
            #stream_ds.attrs['crs'] = self.crs
            #stream_ds.attrs['transform'] = self.transform
            stream_ds.demap.crs = self.crs
            stream_ds.demap.transform = self.transform

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
            val = np.asarray(data_source)[i_list, j_list]

        # stream data
        if isinstance(data_source, xr.Dataset):
            if var_name is None:
                raise ValueError("var_name cannot be None when getting data from dataset")
            else:
                source_dataarray = data_source[var_name]
                i_list, j_list = self.stream_coords_rowcol
                index_list = data_source.demap._index_in_ordered_array(i_list, j_list)
                val = source_dataarray.data[index_list]

        if var_name is not None:
            self._xrobj[var_name] = (('hydro_order'), val)

        return val
    
    def smooth_profile(self, dem: Union[xr.DataArray, np.ndarray]):
        """Return a smoothed channel profile by removing obstacle

        Parameters
        ----------
        dem : Union[xr.DataArray, np.ndarray]
            DEM

        Returns
        -------
        z: array-like
            smoothed channel profile
        """
        z = self.get_value(np.asarray(dem))
        z = z.astype(float)

        for k in range(1, len(z)):
            if z[k] > z[k-1]:
                z[k] = np.nextafter(z[k-1], z[k-1]-1)

        return z

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
        #sub_network_ds.attrs['crs'] = self.crs
        #sub_network_ds.attrs['transform'] = self.transform
        sub_network_ds.demap.crs = self.crs
        sub_network_ds.demap.transform = self.transform

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
        #sub_network_ds.attrs['crs'] = self.crs
        #sub_network_ds.attrs['transform'] = self.transform
        sub_network_ds.demap.crs = self.crs
        sub_network_ds.demap.transform = self.transform

        return sub_network_ds
    
    
    def calculate_chi_ksn(self, dem: Union[xr.DataArray, np.ndarray],
                          drainage_area: Union[xr.DataArray, np.ndarray],
                          ref_concavity=0.45, ref_drainage_area=1.0, ksn_elev_window=20):
        downstream = np.asarray(self._xrobj['downstream'])
        dist_up = np.asarray(self._xrobj['distance_upstream'])
        drainage_area = self.get_value(drainage_area)

        chi = _calculate_chi_impl(downstream, dist_up, drainage_area, ref_concavity, ref_drainage_area)

        elev = self.get_value(dem)

        ksn = _calculate_ksn_impl(downstream, chi, elev, elev_window=ksn_elev_window)

        self._xrobj["chi"] = (["hydro_order"], chi)
        self._xrobj["ksn"] = (["hydro_order"], ksn)
    
        return chi, ksn
    
    def calculate_stream_order(self, method='strahler'):
        downstream = np.asarray(self._xrobj['downstream'])

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

        self._xrobj['stream_order'] = (["hydro_order"], stream_order)
        
        return stream_order
    

    def update_transform(self, transform):
        old_transform = self._xrobj.demap.transform

        relative_transform = ~transform * old_transform

        old_rows, old_cols = np.asarray(self._xrobj['rows']), np.asarray(self._xrobj['cols'])

        new_cols, new_rows = relative_transform * (old_cols, old_rows)

        new_rows = round_rowcol_coord(new_rows)
        new_cols = round_rowcol_coord(new_cols)

        new_stream_ds = self._xrobj.copy(deep=True)

        new_stream_ds['rows'] = ('hydro_order', new_rows)
        new_stream_ds['cols'] = ('hydro_order', new_cols)

        new_stream_ds.demap.transform = transform

        return new_stream_ds
    


###
# Implementation using numba
###

@_speed_up
def _index_in_ordered_array_impl(row, col, ordered_nodes):

    index_list = np.zeros(len(row), dtype=np.int32)
    for i in range(len(index_list)):
        index_list[i] = np.where(np.logical_and(ordered_nodes[:, 0] == row[i], ordered_nodes[:, 1] == col[i]))[0].astype(np.int32)[0]

    return index_list


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

@_speed_up
def _calculate_chi_impl(downstream: np.ndarray,
                        dist_up: np.ndarray,
                        drainage_area: np.ndarray,
                        ref_concavity, ref_drainage_area):
    
    chi = -np.ones_like(downstream, dtype=np.single)

    dchi = np.power(ref_drainage_area/drainage_area, ref_concavity)

    for k in range(len(downstream)-1, -1, -1):
        d_k = downstream[k]
        if d_k != -1:
            d_dist = dist_up[k] - dist_up[d_k]
            chi[k] = chi[d_k] + (dchi[k] + dchi[d_k])/2 * d_dist
        else:
            chi[k] = 0

    return chi

@_speed_up
def _calculate_ksn_impl(downstream: np.ndarray, chi: np.ndarray, elev: np.ndarray, elev_window):
    
    # find the downstream nodes that are within the elevation window
    # indicated by its index elev_down_k
    '''
    elev_down_k = downstream.copy()
    domain, = np.where(elev_down_k > -1) # nodes that have not found a downstream node
    while len(domain) > 0:
        domain, = np.where(
            np.logical_and(
                elev_down_k > -1,
                np.logical_and(
                    downstream[elev_down_k] > -1,
                    elev - elev[downstream[elev_down_k]] < elev_window/2,
                ) 
            )
        )

        elev_down_k[domain] = downstream[elev_down_k[domain]]
    '''
    # this simple version is faster
    elev_down_k = -np.ones_like(downstream)
    for k in range(len(downstream)):
        if downstream[k] != -1:
            down_k = downstream[k]
            while downstream[down_k] != -1 and elev[k]-elev[downstream[down_k]] < elev_window/2:
                down_k = downstream[down_k]
            elev_down_k[k] = down_k
        else:
            elev_down_k[k] = -1
    
    elev_down = elev - elev[elev_down_k]
    elev_down[elev_down_k < 0] = 0 # elev_down is -1 for nodes that have no downstream node
    elev_down[elev_down < 0] = 0 # fix reverse slope
    chi_down = chi - chi[elev_down_k]
    chi_down[elev_down_k < 0] = 0

    ksn_down = np.zeros(len(chi_down))
    ksn_down[chi_down > 0] = elev_down[chi_down > 0]/chi_down[chi_down > 0]

    # find upstream ksn
    # one node may have multiple upstream nodes
    # in this case, we take the average of the upstream ksn weighted by their
    # drainage area

    donor_num = np.ones(len(downstream), dtype=np.int32) # pseudo drainage area
    for k in range(len(downstream)):
        if downstream[k] > -1:
            donor_num[downstream[k]] += donor_num[k]
    
    ksn_up = np.zeros_like(ksn_down)
    count_up = np.zeros(len(ksn_up), dtype=np.int32)
    for k in range(len(downstream)):
        down_k = downstream[k]
        if down_k > -1:
            # for down_k, calculate mean ksn_up weighted by draiange area.
            # count_up records the total drainage area of upstream nodes that
            # has been involvded in calculating ksn_up
            ksn_up[down_k] = (ksn_up[down_k]*count_up[down_k] + ksn_down[k]*donor_num[k])/(count_up[down_k] + donor_num[k])
            count_up[down_k] += donor_num[k]

    ksn = (ksn_down + ksn_up)/2
    ksn = np.where(downstream > -1, ksn, ksn_up)
    ksn = np.where(donor_num > 1, ksn, ksn_down)

    return ksn