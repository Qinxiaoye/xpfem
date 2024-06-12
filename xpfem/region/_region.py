import numpy as np

class Region:
    def __init__(self, mesh,umat,trcounter):
        self.mesh = mesh
        self.umat = umat
        self.ndim = mesh.ndim
        self.mnode = mesh.mnode
        self.trcounter = trcounter
