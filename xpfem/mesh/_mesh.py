import numpy as np

class Mesh():
    def __init__(self, points, cells,mnode,ndim, cell_type=None):
        self.points = np.array(points)
        self.cells = np.array(cells,'int64')
        self.cell_type = cell_type
        self.ndim = ndim
        self.mnode = mnode

        self.sumNode = points.shape[0]
        self.sumElem = cells.shape[0]



    def __repr__(self):
        header = "<xpfem Mesh object>"
        points = f"  Number of points: {len(self.points)}"
        cells_header = "  Number of cells:"
        cells = [f"    {self.cell_type}: {self.ncells}"]

        return "\n".join([header, points, cells_header, *cells])

    def __str__(self):
        return self.__repr__()