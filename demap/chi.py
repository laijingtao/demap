import copy
import numpy as np

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

    """
    drainage_area_data = drainage_area.data
    ordered_nodes = stream_network.ordered_nodes
    downstream = stream_network.downstream
    sn_rowcol_to_xy = stream_network.rowcol_to_xy

    chi = np.zeros(len(ordered_nodes))

    for k in range(len(ordered_nodes)-1, -1, -1):
        if downstream[k] != -1:
            i, j = ordered_nodes[k]
            r_i, r_j = ordered_nodes[downstream[k]]
            x1, y1 = sn_rowcol_to_xy(i, j)
            x2, y2 = sn_rowcol_to_xy(r_i, r_j)
            chi[k] = chi[downstream[k]] + (ref_drainage_area/drainage_area_data[i, j])**ref_concavity\
                * distance_p2p(x1, y1, x2, y2)
    """
    ordered_nodes = stream_network.ordered_nodes
    downstream = stream_network.downstream

    # build a pseudo receiver, # its shape may not be the same as original receiver
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

    chi_grid = _calculate_chi_from_receiver_impl(drainage_area.data,
                                                 pseudo_receiver, ordered_nodes,
                                                 ref_concavity, ref_drainage_area,
                                                 affine_matrix)

    i_list = ordered_nodes[:, 0]
    j_list = ordered_nodes[:, 1]
    chi = chi_grid[i_list, j_list]

    stream_network.attrs['chi'] = chi


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
