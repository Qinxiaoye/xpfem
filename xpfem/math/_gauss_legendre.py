import numpy as np
from ._gaussInt import gaussInt

class gauss_legendre(gaussInt):
    def __init__(self,intOrder:int,dim:int):
        p, w = np.polynomial.legendre.leggauss(intOrder)

        if dim == 1:
            points = p
            weights = w
        elif dim == 2:
            points = np.zeros((intOrder**2,2))
            weights = np.zeros(intOrder**2)
            ID = 0
            for n in range(0,intOrder):
                for m in range(0,intOrder):
                    weights[ID] = w[n]*w[m]
                    points[ID] = np.array([p[n],p[m]])
                    ID = ID+1

        elif dim == 3:
            points = np.zeros((intOrder**3,3))
            weights = np.zeros(intOrder**3)
            ID = 0
            for n in range(0,intOrder):
                for m in range(0,intOrder):
                    for k in range(0,intOrder):
                        weights[ID] = w[n]*w[m]*w[k]
                        points[ID] = np.array([p[n],p[m],p[k]])
                        ID = ID+1

        
        super().__init__(points,weights)
