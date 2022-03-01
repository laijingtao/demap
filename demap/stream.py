import numpy as np
import copy
from typing import Union

from .helpers import rowcol_to_xy, xy_to_rowcol
from ._base import is_verbose, INT
from .geogrid import GeoGrid
from ._impl import _build_ordered_array_impl


class _StreamBase:
    """Base class for stream and stream network"""

    def __init__(self):
        self.ordered_nodes = None
        self.crs = None
        self.transform = None
        self.attrs = {}

    def rowcol_to_xy(self, row, col):
        return rowcol_to_xy(row, col, self.transform)

    def xy_to_rowcol(self, x, y):
        return xy_to_rowcol(x, y, self.transform)

    def _build_hashmap(self):
        assert isinstance(self.ordered_nodes, np.ndarray), 'Missing ordered_nodes or wrong type'

        ordered_nodes = self.ordered_nodes
        nodes_hash = {}
        for k in range(len(ordered_nodes)):
            row, col = ordered_nodes[k]
            nodes_hash['{}_{}'.format(int(row), int(col))] = k

        self._nodes_hash = nodes_hash

    def index_of(self, row, col):
        return self._nodes_hash['{}_{}'.format(int(row), int(col))]

    def nearest_to_xy(self, x, y):
        """
        Return the stream node nearest to given x, y geographic coordinates.
        """
        row, col = self.xy_to_rowcol(x, y)
        d_i = np.abs(self.ordered_nodes[:, 0] - row)
        d_j = np.abs(self.ordered_nodes[:, 1] - col)
        dist = np.sqrt(np.power(d_i, 2) + np.power(d_j, 2))
        node = self.ordered_nodes[np.argmin(dist)]

        return node


class Stream(_StreamBase):
    """A stream"""

    def __init__(self, ordered_nodes, **kwargs):
        self.ordered_nodes = np.array(ordered_nodes, dtype=INT)  # n by 2 np.ndarray
        self._build_hashmap()
        self.crs = kwargs.get('crs', None)
        self.transform = kwargs.get('transform', None)
        if self.transform is not None:
            _ = self.get_upstream_distance()
        else:
            self.dist_up = None
        self.attrs = {}

    def __repr__(self):
        return f'Stream({self.ordered_nodes})'

    def get_upstream_distance(self):
        if self.transform is None:
            raise RuntimeError("No transform info for this stream,\
                cannot calculate the distance")

        if len(self.ordered_nodes) == 1:
            self.dist_up = np.array([0.])
            return self.dist_up

        dist_up = np.zeros(len(self.ordered_nodes))
        dist_down = np.zeros(len(self.ordered_nodes))

        i_list = self.ordered_nodes[:, 0]
        j_list = self.ordered_nodes[:, 1]

        x_list, y_list = self.rowcol_to_xy(i_list, j_list)

        d_x = np.abs(x_list[:-1] - x_list[1:])
        d_y = np.abs(y_list[:-1] - y_list[1:])
        d_dist = np.sqrt(np.power(d_x, 2) + np.power(d_y, 2))

        dist_down[1:] = np.cumsum(d_dist)
        dist_up = dist_down[-1] - dist_down

        self.dist_up = dist_up
        return self.dist_up

    def get_value(self, data_source: Union[GeoGrid, 'StreamNetwork'], attr_name=None):
        if not isinstance(data_source, (GeoGrid, StreamNetwork)):
            raise TypeError("Unsupported data_source type")

        if isinstance(data_source, GeoGrid):
            i_list = self.ordered_nodes[:, 0]
            j_list = self.ordered_nodes[:, 1]
            val = data_source.data[i_list, j_list]

        if isinstance(data_source, StreamNetwork):
            if attr_name is None:
                raise ValueError("attr_name cannot be None when getting data from StreamNetwork")
            else:
                index_of = data_source.index_of
                val = np.zeros(len(self.ordered_nodes))
                for k in range(len(self.ordered_nodes)):
                    i, j = self.ordered_nodes[k]
                    val[k] = data_source.attrs[attr_name][index_of(i, j)]

        if attr_name is not None:
            self.attrs[attr_name] = val

        return val

    def dir_vector_at_rowcol(self, row, col, smooth_range=1e3):
        idx = self.index_of(row, col)
        dist_up = self.dist_up
        in_range = np.abs(dist_up - dist_up[idx]) <= smooth_range*0.5

        k = idx
        while k > 0 and in_range[k]:
            k -= 1
        row_up, col_up = self.ordered_nodes[k]

        k = idx
        while k < len(self.ordered_nodes) - 1 and in_range[k]:
            k += 1
        row_down, col_down = self.ordered_nodes[k]

        x_up, y_up = self.rowcol_to_xy(row_up, col_up)
        x_down, y_down = self.rowcol_to_xy(row_down, col_down)

        dir_vector = np.array([x_down - x_up, y_down - y_up])

        return dir_vector


class StreamNetwork(_StreamBase):
    """
    A StreamNetwork object saves the stream network using link information

    `ordered_nodes` saves the coordinates of all the nodes in this network,
    with upstream-to-downstream order.

    `downstream` and `upstream` save the link information. Both of them are
    saved as indexes in the `ordered_nodes`.

    `downstream` saves receiver info, `downstream[i]` is the receiver of `ordered_noded[i]`
    `upstream` saves donor info, `upstream[i]` is the donor of `ordered_noded[i]`
        `upstream` is a n x 9 np.ndarray, `upstream[i]` represents the number of
        donors plus 8 possilbe donor.

    if `downstream` is -1, then this is an outlet.
    if `
    """

    def __init__(self, receiver: GeoGrid = None, **kwargs):
        if receiver is not None:
            self.build_from_receiver_array(receiver)
        elif all(k in kwargs for k in ('ordered_nodes', 'downstream', 'upstream')):
            self.ordered_nodes = kwargs['ordered_nodes']
            self.n_nodes = len(self.ordered_nodes)
            self._build_hashmap()
            self.downstream = kwargs['downstream']
            self.upstream = kwargs['upstream']
            self.crs = kwargs.get('crs', None)
            self.transform = kwargs.get('transform', None)
        else:
            print("Warning: incomplete input, an empty StreamNetwork was created")

        self.attrs = {}

    def build_from_receiver_array(self, receiver: GeoGrid):
        self.transform = copy.deepcopy(receiver.transform)
        self.crs = copy.deepcopy(receiver.crs)

        self.ordered_nodes = _build_ordered_array_impl(receiver.data)
        self.n_nodes = len(self.ordered_nodes)
        self._build_hashmap()

        ordered_nodes = self.ordered_nodes
        i_list = ordered_nodes[:, 0]
        j_list = ordered_nodes[:, 1]
        # receiver_list follow the order of ordered_nodes
        receiver_list = receiver.data[i_list, j_list]

        downstream = -np.ones(self.n_nodes, dtype=INT)
        upstream = -np.ones((self.n_nodes, 9), dtype=INT)
        upstream[:, 0] = 0

        index_of = self.index_of
        for k in range(len(receiver_list)):
            from_idx = index_of(ordered_nodes[k, 0], ordered_nodes[k, 1])
            to_idx = index_of(receiver_list[k, 0], receiver_list[k, 1])

            if from_idx != to_idx:  # not outlet
                downstream[from_idx] = to_idx
                upstream[to_idx, 0] += 1  # number of upstream nodes
                upstream[to_idx, upstream[to_idx, 0]] = from_idx

        self.ordered_nodes = ordered_nodes
        self.downstream = downstream
        self.upstream = upstream

    def to_streams(self, mode='all'):
        if is_verbose():
            print("Converting stream network to streams ...")

        if mode not in ['all', 'tributary']:
            raise ValueError("Unknow mode, accepted modes are: \'all\', \'tributary\'")

        ordered_nodes = self.ordered_nodes
        upstream = self.upstream
        downstream = self.downstream
        index_of = self.index_of

        # streams_idx[k] represents stream section starting from ordered_nodes[k]
        # recorded by index in ordered_nodes
        streams_idx = np.empty(self.n_nodes, dtype=object)

        for k in range(self.n_nodes-1, -1, -1):
            if downstream[k] == -1:
                streams_idx[k] = np.array([k])
            else:
                r_i, r_j = ordered_nodes[downstream[k]]
                streams_idx[k] = np.append(k, streams_idx[index_of(r_i, r_j)])

        # we only want stream start from the head
        streams_idx = streams_idx[np.where(upstream[:, 0] == 0)]

        # build streams from stream_idx
        streams = np.empty(len(streams_idx), dtype=object)
        for k in range(len(streams_idx)):
            streams[k] = Stream(ordered_nodes=ordered_nodes[streams_idx[k]],
                                crs=self.crs, transform=self.transform)

        # sort by length
        length_list = np.array([st.dist_up[0] for st in streams])
        sort_idx = np.argsort(length_list)[::-1]
        streams = streams[sort_idx]
        streams_idx = streams_idx[sort_idx]  # this is for split tributaries

        # split the stream network into tributaries
        if mode == 'tributary':
            in_streams = np.zeros(self.n_nodes, dtype=bool)

            for k in range(len(streams_idx)):
                not_in_streams = ~in_streams[streams_idx[k]]
                if streams_idx[k][not_in_streams][-1] != streams_idx[k][-1]:
                    # this is a tributary, add junction node
                    junction = not_in_streams.nonzero()[0][-1] + 1
                    not_in_streams[junction] = True
                streams[k].ordered_nodes = streams[k].ordered_nodes[not_in_streams]
                streams[k].dist_up = streams[k].dist_up[not_in_streams]

                # new coords, put them into network
                added_nodes_idx = streams_idx[k][not_in_streams]
                in_streams[added_nodes_idx] = True

            # sort by length again, note it's the length of tributary stream
            length_list = np.array([st.dist_up[0]-st.dist_up[-1] for st in streams])
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
        downstream = self.downstream
        ordered_nodes = self.ordered_nodes
        sub_mask = np.zeros(len(ordered_nodes), dtype=np.int8)

        k = index_of(row, col)
        sub_mask[k] = True

        # remove all upstream nodes first
        new_upstream = copy.deepcopy(self.upstream)
        new_upstream[k] = -np.ones(9, dtype=INT)
        new_upstream[k, 0] = 0

        # go downstream
        while downstream[k] != -1:
            i, j = ordered_nodes[downstream[k]]
            k = index_of(i, j)
            sub_mask[k] = True

        new_ordered_nodes = copy.deepcopy(ordered_nodes[np.where(sub_mask)])
        new_downstream = copy.deepcopy(downstream[np.where(sub_mask)])
        new_upstream = new_upstream[np.where(sub_mask)]

        # update index in new_dowstream and new_upstream
        new_idx = -np.ones(len(ordered_nodes))
        new_idx[np.where(sub_mask)] = np.arange(len(new_ordered_nodes))
        # new_idx is essentially a function that convert old index to new index
        new_downstream[np.where(new_downstream >= 0)] =\
            new_idx[new_downstream[np.where(new_downstream >= 0)]]
        new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)] =\
            new_idx[new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)]]

        sub_network = StreamNetwork(ordered_nodes=new_ordered_nodes,
                                    downstream=new_downstream,
                                    upstream=new_upstream)

        sub_network.crs = self.crs
        sub_network.transform = self.transform

        # extract attrs
        for key in self.attrs:
            sub_network.attrs[key] = copy.deepcopy(self.attrs[key][np.where(sub_mask)])

        return sub_network

    def _extract_from_rowcol_up(self, row, col):
        index_of = self.index_of
        upstream = self.upstream
        ordered_nodes = self.ordered_nodes
        sub_mask = np.zeros(len(ordered_nodes), dtype=np.int8)

        k = index_of(row, col)
        sub_mask[k] = True

        # remove all downstream nodes first
        new_downstream = copy.deepcopy(self.downstream)
        new_downstream[k] = -1

        # use a queue to build sub_mask
        from collections import deque
        q = deque([k], maxlen=len(ordered_nodes)+10)

        while len(q) > 0:
            k = q.popleft()
            for up_node in upstream[k, 1:upstream[k, 0]+1]:
                sub_mask[up_node] = True
                q.append(up_node)

        new_ordered_nodes = copy.deepcopy(ordered_nodes[np.where(sub_mask)])
        new_downstream = new_downstream[np.where(sub_mask)]
        new_upstream = copy.deepcopy(upstream[np.where(sub_mask)])

        # update index in new_dowstream and new_upstream
        new_idx = -np.ones(len(ordered_nodes))
        new_idx[np.where(sub_mask)] = np.arange(len(new_ordered_nodes))
        # new_idx is essentially a function that convert old index to new index
        new_downstream[np.where(new_downstream >= 0)] = \
            new_idx[new_downstream[np.where(new_downstream >= 0)]]
        new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)] = \
            new_idx[new_upstream[:, 1:][np.where(new_upstream[:, 1:] >= 0)]]

        sub_network = StreamNetwork(ordered_nodes=new_ordered_nodes,
                                    downstream=new_downstream,
                                    upstream=new_upstream)

        sub_network.crs = self.crs
        sub_network.transform = self.transform

        # extract attrs
        for key in self.attrs:
            sub_network.attrs[key] = copy.deepcopy(self.attrs[key][np.where(sub_mask)])

        return sub_network


def _merge_network(network1: StreamNetwork, network2: StreamNetwork):
    # TODO
    raise NotImplementedError
