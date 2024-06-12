import numpy as np
from xpfem import shapefunction

class GlobalK2D:
    def __init__(self, mnode):
        self.Dfun= shapefunction.Dfun(mnode)
        self.mnode = mnode

    def GK(self, region): # region.mesh.cells, region.mesh.points, region.umat
        elem = region.mesh.cells
        node = region.mesh.points
        umat = region.umat
        

        sumElem = elem.shape[0]

        nnz0 = (self.mnode*2)*(self.mnode*2)
        nnz = sumElem*(self.mnode*2)*(self.mnode*2)

        
        u = np.zeros(nnz,dtype='int64')
        v = np.zeros(nnz,dtype='int64')
        a = np.zeros(nnz)
        x = np.zeros((sumElem, self.mnode))
        y = np.zeros((sumElem, self.mnode))
        for m in range(1, self.mnode+1):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
        # 按照单元进行循环
        ID = 0
        for name in umat.nameList:
            D = umat.part[name].hessian()
            
            for elemID in umat.set[name]:
                ID = ID+1
                K = self.elemK2D(x[elemID, :], y[elemID, :], D)

                nodeID = elem[elemID, :]
                ndfID = np.zeros(2 * self.mnode)
                ndfID[0::2] = nodeID * 2
                ndfID[1::2] = nodeID * 2+1
                ddd = np.tile(ndfID, (2 * self.mnode, 1))
                u[nnz0*(ID-1):nnz0*ID] = ddd.flatten('F')
                v[nnz0*(ID-1):nnz0*ID] = np.tile(ndfID, 2 * self.mnode)
                a[nnz0*(ID-1):nnz0*ID] = K.flatten('F')
                # u = np.concatenate((u, ddd.flatten('F')))
                # v = np.concatenate((v,  np.tile(ndfID, 2 * self.mnode)))
                # a = np.concatenate((a, K.flatten('F')))
        return u, v, a

    def elemK2D(self, x, y, D):
        Ke = np.zeros((self.mnode * 2, self.mnode * 2))


        if self.mnode == 8:
            nip = 3
            ks = np.array([-0.774596669241483, 0, 0.774596669241483])
            w = np.array([0.55555555555556, 0.888888888888888889, 0.55555555555556])
        else:
            nip = 2
            ks = np.array([-0.577350269189626, 0.577350269189626])
            w = np.array([1, 1])

        for m in range(1, nip+1):
            for n in range(1, nip+1):
                J, B = self.elemB2D(x, y, ks[n - 1], ks[m - 1])
                Ke = Ke + w[m - 1] * w[n - 1] * np.dot(np.dot(B.T, D), B) * np.linalg.det(J)
        return Ke

    def elemB2D(self, x, y, ks, yt):
        B = np.zeros((3, 2 * self.mnode))
        N_ks, N_yt = self.Dfun.dfun2D(ks, yt)
        NksNyt = np.concatenate((N_ks.T, N_yt.T))
        xy = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)))
        J = np.dot(NksNyt, xy.T)
        for n in range(1, self.mnode+1):
            N_cor = np.linalg.solve(J, NksNyt[:, n-1])
            B[0, 2 * n - 2] = N_cor[0]
            B[1, 2 * n - 1] = N_cor[1]
            B[2, 2 * n - 1] = N_cor[0]
            B[2, 2 * n - 2] = N_cor[1]
        return J, B
