import copy
import numpy as np
from typing import Union

from .geogrid import GeoGrid
from .stream import Stream, StreamNetwork
from ._impl import (_calculate_chi_from_receiver_impl,
                    _calculate_dist_up_impl,
                    _build_pseudo_receiver_from_network_impl)
from ._base import is_verbose, INT, _speed_up
from .helpers import transform_to_ndarray


def calculate_chi(stream_network: StreamNetwork, drainage_area: GeoGrid, 
                  ref_concavity=0.45, ref_drainage_area=1.0, **kwargs):
    """
    Calculate chi and save it in the given StreamNetwork.
    """
    if is_verbose():
        print("Calculating chi ...")

    ordered_nodes = stream_network._ordered_nodes()
    downstream = stream_network.dataset['downstream'].data

    pseudo_receiver = _build_pseudo_receiver_from_network_impl(
        ordered_nodes, downstream)

    dist_up_grid = _calculate_dist_up_impl(
        pseudo_receiver, ordered_nodes, stream_network.dx(), stream_network.dy())

    if ref_concavity == 'auto':
        try:
            dem: GeoGrid = kwargs['dem']
        except:
            raise KeyError("A DEM is needed for auto concavity method.")
        concavity_range = kwargs.get('concavity_range', [0.3, 0.7])
        chi_grid, best_fit_theta, theta_list, misfit_list = _calculate_chi_auto_concavity(
            stream_network, dem.dataarray.data, pseudo_receiver, ordered_nodes,
            dist_up_grid, drainage_area.dataarray.data, ref_drainage_area, concavity_range=concavity_range)
    else:
        chi_grid = _calculate_chi_from_receiver_impl(
            pseudo_receiver, ordered_nodes, dist_up_grid, drainage_area.dataarray.data,
            ref_concavity, ref_drainage_area)

    stream_network.get_value(chi_grid, var_name='chi')

    if ref_concavity == 'auto':
        return stream_network, best_fit_theta, theta_list, misfit_list
    else:
        return stream_network


def calculate_chi_grid(receiver: GeoGrid, ordered_nodes: np.ndarray, drainage_area: GeoGrid, 
                       ref_concavity=0.45, ref_drainage_area=1, **kwargs):

    if is_verbose():
        print("Calculating chi ...")

    dx = np.abs(receiver.dataarray.attrs['transform'][0])
    dy = np.abs(receiver.dataarray.attrs['transform'][4])
    dist_up = _calculate_dist_up_impl(receiver.dataarray.data, ordered_nodes, dx, dy)

    chi_data = _calculate_chi_from_receiver_impl(receiver.dataarray.data, ordered_nodes,
                                                 dist_up, drainage_area.dataarray.data,
                                                 ref_concavity, ref_drainage_area)

    chi = GeoGrid(chi_data, 
                  copy.deepcopy(receiver.dataarray.attrs['crs']),
                  copy.deepcopy(receiver.dataarray.attrs['transform']),
                  copy.deepcopy(receiver.dataarray.attrs['metadata']),
                  nodata=-1)

    return chi


def _calculate_chi_auto_concavity(stream_network: StreamNetwork,
                                  dem_data: np.ndarray,
                                  receiver_data: np.ndarray,
                                  ordered_nodes: np.ndarray,
                                  dist_up_grid: np.ndarray,
                                  drainage_area_data: np.ndarray,
                                  ref_drainage_area,
                                  concavity_range):
    """Implementation of quasi -  Mudd et al., 2018
    (https://esurf.copernicus.org/articles/6/505/2018/)
    """

    if is_verbose():
        print("Finding the best-fit concavity ...")

    stream_list = stream_network.to_streams(mode='tributary')

    misfit_estimator = _least_square_estimate
    best_fit = np.argmin

    theta_list = np.arange(concavity_range[0], concavity_range[1]+1e-5, 0.01)
    misfit_list = np.zeros(len(theta_list))
    for k in range(len(theta_list)):
        chi_grid = _calculate_chi_from_receiver_impl(
            receiver_data, ordered_nodes, dist_up_grid, drainage_area_data,
            theta_list[k], ref_drainage_area)

        misfit = misfit_estimator(stream_list, chi_grid, dem_data)
        misfit_list[k] = misfit

    k = best_fit(misfit_list)
    chi_grid = _calculate_chi_from_receiver_impl(
        receiver_data, ordered_nodes, dist_up_grid, drainage_area_data,
        theta_list[k], ref_drainage_area)

    return chi_grid, theta_list[k], theta_list, misfit_list


def _least_square_estimate(stream_list: np.ndarray, chi_grid: np.ndarray, dem_data: np.ndarray):

    trunk: Stream = stream_list[0]
    z = trunk.get_value(dem_data)

    # normalize the dem by trunk elevation range
    norm_dem_data = (dem_data - z.min()) / (z.max() - z.min())

    z = trunk.get_value(norm_dem_data)
    chi = trunk.get_value(chi_grid)

    from scipy import interpolate
    z_trunk = interpolate.interp1d(chi, z, fill_value="extrapolate")

    misfit = 0
    for s in stream_list[1:]:
        z = s.get_value(norm_dem_data)
        chi = s.get_value(chi_grid)

        residual = np.power(z - z_trunk(chi), 2)
        misfit += np.sum(residual)

    return misfit


def calculate_ksn(dem: GeoGrid, drainage_area: GeoGrid,
                  stream_network: StreamNetwork, ref_concavity=0.45, **kwargs):
    if is_verbose():
        print("Calculating ksn ...")

    '''
    stream_list = stream_network.to_streams('tributary')
    index_of = stream_network.index_of

    A_theta = np.power(drainage_area.data, ref_concavity)

    ksn = np.zeros(len(stream_network.ordered_nodes))
    for s in stream_list:
        s_ordered_nodes = s.ordered_nodes
        z = s.smooth_profile(dem)
        dist_up = s.dist_up
        slope = (z[:-1] - z[1:]) / (dist_up[:-1] - dist_up[1:])
        ksn_s = slope * s.get_value(A_theta)[:-1]
        for k in range(len(ksn_s)):
            row, col = s_ordered_nodes[k]
            ksn[index_of(row, col)] = ksn_s[k]

    '''
    downstream = stream_network.dataset['downstream'].data
    dist_up = stream_network.dataset['distance_upstream'].data

    z = stream_network.smooth_profile(dem)

    A_theta = np.power(stream_network.get_value(drainage_area), ref_concavity)

    slope = np.zeros(len(z))

    not_outlet = downstream != -1
    slope[not_outlet] = (z[not_outlet] - z[downstream[not_outlet]]) \
        / (dist_up[not_outlet] - dist_up[downstream[not_outlet]])

    ksn = slope * A_theta

    stream_network.dataset['ksn'] = (('flow_order'), ksn)

    return stream_network


def calculate_channel_slope(stream_network: Union[StreamNetwork, Stream], elev_window=20, **kwargs):
    if is_verbose():
        print("Calculating channel slope ...")

    if 'elev' in stream_network.dataset:
        elev = stream_network.dataset['elev'].data
    elif 'elevation' in stream_network.dataset:
        elev = stream_network.dataset['elevation'].data
    elif 'dem' in kwargs:
        elev = stream_network.get_value(kwargs['dem'])
    else:
        raise ValueError("Missing elevation information. Pass a dem or add elevation info to the stream network")
    
    flow_order = stream_network.dataset['flow_order'].data

    if isinstance(stream_network, StreamNetwork):
        downstream = stream_network.dataset['downstream'].data
    elif isinstance(stream_network, Stream):
        downstream = flow_order + 1
        downstream[-1] = -1
    

    '''
    downstream_elev_drop = np.zeros_like(downstream)

    for k in flow_order:
        if downstream[k] != -1:
            down_k = downstream[k]
            while downstream[down_k] != -1 and elev[k]-elev[downstream[down_k]] < elev_window/2:
                down_k = downstream[down_k]
            downstream_elev_drop[k] = down_k
        else:
            downstream_elev_drop[k] = -1
    '''
    downstream_elev_drop = downstream.copy()

    domain, = np.where(downstream_elev_drop > 0)
    while len(domain) > 0:
        domain, = np.where(
            np.logical_and(
                downstream_elev_drop > 0,
                np.logical_and(
                    downstream[downstream_elev_drop] > 0,
                    elev[flow_order] - elev[downstream[downstream_elev_drop]] < elev_window/2,
                ) 
            )
        )

        downstream_elev_drop[domain] = downstream[downstream_elev_drop[domain]]

    donor_num = np.zeros(len(flow_order), dtype=int) # pseudo drainage area
    for k in flow_order:
        if downstream[k] > -1:
            donor_num[downstream[k]] += donor_num[k]

    dist = stream_network.dataset['distance_upstream'].data

    elev_down = elev - elev[downstream_elev_drop]
    elev_down[downstream_elev_drop < 0] = 0
    dist_down = dist - dist[downstream_elev_drop]
    dist_down[downstream_elev_drop < 0] = 0

    elev_up = np.zeros_like(elev)
    dist_up = np.zeros_like(dist)
    count_up = np.zeros(len(elev_up), dtype=int)
    for k in flow_order:
        down_k = downstream[k]
        if down_k > -1:
            if count_up[down_k] == 0:
                elev_up[down_k] = elev[k] - elev[down_k]
                dist_up[down_k] = dist[k] - dist[down_k]
                count_up[down_k] = donor_num[k]
            elif donor_num[k] > count_up[down_k]:
                elev_up[down_k] = elev[k] - elev[down_k]
                dist_up[down_k] = dist[k] - dist[down_k]
                count_up[down_k] = donor_num[k]

    slope = (elev_down + elev_up)/(dist_down + dist_up)

    stream_network.dataset['slope'] = (('flow_order'), slope)

    return stream_network