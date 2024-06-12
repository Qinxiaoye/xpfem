import numpy as np


class Element:
    def __init__(self, shape):
        self.shape = shape
        self.dim = self.shape[1]