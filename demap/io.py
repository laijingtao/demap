import rasterio as rio

from .geoarray import GeoArray


def load(filename):
    """Read a geospatial data"""

    with rio.open(filename) as dataset:
        data = dataset.read(1)  # DEM data only have 1 band.
        crs = dataset.crs
        transform = dataset.meta['transform']
        metadata = dataset.meta

    outdata = GeoArray(data, crs, transform, metadata)

    return outdata
