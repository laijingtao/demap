import rasterio as rio

from .geogrid import GeoGrid
from .stream import StreamNetwork


def load(filename):
    """Read a geospatial data"""

    with rio.open(filename) as dataset:
        data = dataset.read(1)  # DEM data only have 1 band.
        crs = dataset.crs
        transform = dataset.meta['transform']
        metadata = dataset.meta

    outdata = GeoGrid(data, crs, transform, metadata)

    return outdata


def network_to_shp(stream_network: StreamNetwork,
                   filename: str):

    import shapefile
    stream_list = stream_network.to_streams(mode='tributary')
    line_list = []
    for stream in stream_list:
        i_list = stream.ordered_nodes[:, 0]
        j_list = stream.ordered_nodes[:, 1]

        x_list, y_list = stream.rowcol_to_xy(i_list, j_list)

        xy_coords = [[x_list[k], y_list[k]] for k in range(len(x_list))]

        line_list.append(xy_coords)

    with shapefile.Writer(filename) as w:
        w.field('name', 'C')
        for line in line_list:
            w.line([line])
            w.record('Stream network generated by DEMAP')
