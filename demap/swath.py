import numpy as np


class Swath:

    def __init__(self, z, dist, end_coords):
        self.z = z
        self.dist = dist
        self.end_coords = end_coords
