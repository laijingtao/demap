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
    pass