import numpy as np
import xarray as xr
from typing import Union

from ._base import _speed_up, _XarrayAccessorBase

class _DEMAccessor(_XarrayAccessorBase):
    
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

    #####################################
    # DEM processing methods
    #####################################

    def get_flow_direction(self, base_level=-9999):

        dem = self._xrobj['dem']

        dem_data = np.asarray(dem)
        dem_data[dem_data == dem.rio.nodata] = -32768

        dem_cond = _fill_depression(dem_data, -32768, self.transform)
        dem_cond = np.where(dem_cond > base_level, dem_cond, -32768)
        flow_dir_data = _flow_dir_from_richdem(dem_cond, -32768, self.transform)

        self._xrobj['dem_cond'] = (('y', 'x'), dem_cond)
        self._xrobj['dem_cond'] = self._xrobj['dem_cond'].rio.write_nodata(dem.rio.nodata)
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


    def process_dem(self, base_level=-9999):
        _ = self.get_flow_direction(base_level=base_level)
        _ = self.build_hydro_order()
        _ = self.accumulate_flow()



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
    dem_rd = dem_rd.astype(dtype=np.float32)

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

    return drainage_area.astype(np.float32)


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