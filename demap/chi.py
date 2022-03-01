import copy
import numpy as np
import numpy.typing as npt

from .geogrid import GeoGrid
from .stream import Stream, StreamNetwork
from ._impl import _calculate_chi_from_receiver_impl
from ._base import is_verbose, INT
from .helpers import transform_to_ndarray


def calculate_chi(drainage_area: GeoGrid, stream_network: StreamNetwork,
                  ref_concavity=0.45, ref_drainage_area=1e6, **kwargs):
    """
    Calculate chi and save it in the given StreamNetwork.
    """
    if is_verbose():
        print("Calculating chi ...")

    ordered_nodes = stream_network.ordered_nodes
    downstream = stream_network.downstream

    # build a pseudo receiver, its shape may not be the same as original receiver
    pseudo_ni = np.max(ordered_nodes[:, 0]) + 1
    pseudo_nj = np.max(ordered_nodes[:, 1]) + 1
    pseudo_receiver = -np.ones((pseudo_ni, pseudo_nj, 2), dtype=INT)

    for k in range(len(ordered_nodes)):
        i, j = ordered_nodes[k]
        if downstream[k] != -1:
            r_i, r_j = ordered_nodes[downstream[k]]
        else:
            r_i, r_j = i, j
        pseudo_receiver[i, j] = [r_i, r_j]

    affine_matrix = transform_to_ndarray(stream_network.transform)

    if ref_concavity == 'auto':
        try:
            dem: GeoGrid = kwargs['dem']
        except:
            raise KeyError("A DEM is needed for auto concavity method.")
        concavity_range = kwargs.get('concavity_range', [0.3, 0.7])
        chi_grid, theta_list, misfit_list = _calculate_chi_auto_concavity(
            stream_network.to_streams(mode='tributary'), dem.data,
            drainage_area.data, pseudo_receiver, ordered_nodes,
            ref_drainage_area, concavity_range=concavity_range)
    else:
        chi_grid = _calculate_chi_from_receiver_impl(
            drainage_area.data, pseudo_receiver, ordered_nodes,
            ref_concavity, ref_drainage_area, affine_matrix)

    i_list = ordered_nodes[:, 0]
    j_list = ordered_nodes[:, 1]
    chi = chi_grid[i_list, j_list]

    stream_network.attrs['chi'] = chi

    if ref_concavity == 'auto':
        return theta_list, misfit_list


def calculate_chi_grid(drainage_area: GeoGrid, receiver: GeoGrid,
                       ordered_nodes: np.ndarray,
                       ref_concavity=0.45, ref_drainage_area=1e6, **kwargs):

    if is_verbose():
        print("Calculating chi ...")

    affine_matrix = transform_to_ndarray(drainage_area.transform)

    chi_data = _calculate_chi_from_receiver_impl(drainage_area.data,
                                                 receiver.data, ordered_nodes,
                                                 ref_concavity, ref_drainage_area,
                                                 affine_matrix)

    chi = GeoGrid(chi_data, copy.deepcopy(receiver.crs),
                  copy.deepcopy(receiver.transform),
                  copy.deepcopy(receiver.metadata), nodata=-1)

    return chi


def _calculate_chi_auto_concavity(stream_list: np.ndarray,
                                  dem_data: np.ndarray,
                                  drainage_area_data: np.ndarray,
                                  receiver_data: np.ndarray,
                                  ordered_nodes: np.ndarray,
                                  ref_drainage_area,
                                  concavity_range):
    """Implementation of quasi -  Mudd et al., 2018
    (https://esurf.copernicus.org/articles/6/505/2018/)
    """

    if is_verbose():
        print("Finding the best-fit concavity ...")

    affine_matrix = transform_to_ndarray(stream_list[0].transform)

    theta_list = np.arange(concavity_range[0], concavity_range[1]+1e-5, 0.01)
    misfit_list = np.zeros(len(theta_list))
    for k in range(len(theta_list)):
        chi_grid = _calculate_chi_from_receiver_impl(
            drainage_area_data, receiver_data, ordered_nodes,
            theta_list[k], ref_drainage_area, affine_matrix)

        misfit = _estimate_misfit(stream_list, chi_grid, dem_data)
        misfit_list[k] = misfit

    k = np.argmin(misfit_list)
    chi_grid = _calculate_chi_from_receiver_impl(
        drainage_area_data, receiver_data, ordered_nodes,
        theta_list[k], ref_drainage_area, affine_matrix)

    return chi_grid, theta_list, misfit_list


def _estimate_misfit(stream_list: np.ndarray, chi_grid: np.ndarray, dem_data: np.ndarray):

    trunk: Stream = stream_list[0]
    z = trunk.get_value(dem_data)
    chi = trunk.get_value(chi_grid)

    from scipy import interpolate
    z_trunk = interpolate.interp1d(chi, z, fill_value="extrapolate")

    misfit = 0
    for s in stream_list[1:]:
        z = s.get_value(dem_data)
        chi = s.get_value(chi_grid)

        residual = np.abs(z - z_trunk(chi))
        misfit += np.sum(residual)

    return misfit
