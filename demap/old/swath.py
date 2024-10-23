import numpy as np


class Swath:

    def __init__(self, z, dist, end_coords, border_coords=None):
        self.z = z
        self.dist = dist
        self.end_coords = end_coords
        self.border_coords = border_coords
