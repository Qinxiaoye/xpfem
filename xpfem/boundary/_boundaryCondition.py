import numpy as np
import scipy as sci
from ._Direct import Direct

class boundaryCondition(Direct):
    def __init__(self):
        pass

    def boundary_1(self, GK_u, GK_v, GK_a, nodeForce, fixNode, sizeK):

        a = np.arange(0, sizeK)
        c = np.ones(sizeK)
        c[fixNode[:]] = 0
        I = sci.sparse.coo_matrix((np.ones(sizeK), (a, a)), shape=(sizeK, sizeK)).tocsr()
        b = sci.sparse.coo_matrix((c, (a, a)), shape=(sizeK, sizeK)).tocsr()
        GK = sci.sparse.coo_matrix((GK_a, (GK_u, GK_v)), shape=(sizeK, sizeK)).tocsr()
        GK = b * GK * b
        GK = GK - (b - I)

        if sci.sparse.issparse(nodeForce):
            force = b*nodeForce
        else:
            force = b*sci.sparse.csr_matrix(nodeForce.reshape(-1, 1))

        return GK, force
    
    