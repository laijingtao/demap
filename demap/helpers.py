import math
import os
import sys
import threading
import time
import numpy as np
from rasterio import Affine

def rowcol_to_xy(row, col, transform: Affine):
    """
    Returns geographic coordinates given GeoArray data (row, col) coordinates.

    This function will return the coordinates of the center of the pixel.
    """
    # offset for center
    row = np.asarray(row)
    col = np.asarray(col)
    row_off = 0.5
    col_off = 0.5
    return transform * (col+col_off, row+row_off)


def xy_to_rowcol(x, y, transform: Affine):
    """
    Returns GeoArray data (row, col) coordinates of the pixel that contains
    the given geographic coordinates (x, y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    col, row = ~transform * (x, y)
    col = np.floor(col).astype(dtype=np.int32)
    row = np.floor(row).astype(dtype=np.int32)
    return row, col


def xy_to_latlon(x, y, crs):
    import pyproj

    x = np.asarray(x)
    y = np.asarray(y)
    
    proj_crs = pyproj.crs.CRS.from_wkt(crs.to_wkt())
    p = pyproj.Proj(proj_crs)
    
    lon, lat = p(x, y, inverse=True)
    
    return lat, lon

def latlon_to_xy(lat, lon, crs):
    import pyproj

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    proj_crs = pyproj.crs.CRS.from_wkt(crs.to_wkt())
    p = pyproj.Proj(proj_crs)
    
    x, y = p(lon, lat)
    
    return x, y


def transform_to_ndarray(transform: Affine) -> np.ndarray:
    a = transform
    b = np.zeros((3, 3))
    b[0, :] = a[:3]
    b[1, :] = a[3:6]
    b[2, :] = [0, 0, 1]
    return b


def distance_p2p(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))


class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
    """
    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char