# ============================================================
# Implementation. These methods can use `numba` to speed up.
# ============================================================
# Currently all these functions set default nodata [-1, -1] for receiver.
# Plan to enable a way to pass a nodata value for these funcs in the future.

import numpy as np

from ._base import INT, USE_NUMBA


def _speed_up(func):
    """A conditional decorator that use numba to speed up the function"""
    if USE_NUMBA:
        import numba
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
