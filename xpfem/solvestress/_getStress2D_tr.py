import numpy as np
from xpfem import shapefunction, math


class getStress2D_tr:
    def __init__(self, mnode):
        self.Dfun = shapefunction.Dfun_tr(mnode)
        self.mnode = mnode

    def evualStress(self, displacement, region):
        elem = region.mesh.cells
        node = region.mesh.points
        displacement = displacement.reshape(-1, 2)
        sumElem = elem.shape[0]
        sumNode = node.shape[0]
        Bmat, Dmat = self.evualMassB(region)
        s = np.zeros((sumElem * 9, 1))
        stress = np.zeros((sumNode, 3))
        nodeMes = np.zeros((sumNode, 1))
        dbu = np.zeros((9, 1))
        for n in range(sumElem):
            elemU = displacement[elem[n, :], :]
            elemU = elemU.reshape(-1, 1)
            Bu = np.dot(Bmat[9 * n:9 * (n + 1), :], elemU)
            for m in range(3):
                dbu[3 * m:3 * m + 3] = np.dot(Dmat[n * 3:n * 3 + 3, :], Bu[m * 3:m * 3 + 3])
            s[9 * n: 9 * (n + 1)] = dbu
        s = s.reshape(-1, 3)
        for n in range(sumElem):
            nodeMes[elem[n, :]] += 1  # 节点被几个单元共用
            stress[elem[n, 0:3], :] += s[3 * n:3 * (n + 1), :]
        stress = stress / nodeMes
        if self.mnode == 8:
            stress[elem[:, 4], :] = (stress[elem[:, 0], :] + stress[elem[:, 1], :]) / 2
            stress[elem[:, 5], :] = (stress[elem[:, 1], :] + stress[elem[:, 2], :]) / 2
            stress[elem[:, 6], :] = (stress[elem[:, 2], :] + stress[elem[:, 3], :]) / 2
            stress[elem[:, 7], :] = (stress[elem[:, 3], :] + stress[elem[:, 0], :]) / 2
        a = (stress[:, 0] + stress[:, 1]) / 2
        b = ((stress[:, 0] - stress[:, 1]) / 2) ** 2 + stress[:, 2] ** 2
        smax = a + np.sqrt(b)
        smin = a - np.sqrt(b)
        mises = np.sqrt(smax ** 2 + smin ** 2 + (smax - smin) ** 2 / 2)
        return mises

    def evualMassB(self, region):
        elem = region.mesh.cells
        node = region.mesh.points
        umat = region.umat

        sumElem = elem.shape[0]

        nnz0 = (self.mnode * 2) * (self.mnode * 2)
        x = np.zeros((sumElem, self.mnode))
        y = np.zeros((sumElem, self.mnode))
        for m in range(1, self.mnode + 1):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]

        # 高斯积分,二维的固定为2阶，理论上四节点可以使用一阶
        self.hammer_int = math.hammer(intOrder=2, dim=2)
        nge = len(self.hammer_int.points) * 3
        Bmat = np.zeros((nge * sumElem, self.mnode * 2))
        Dmat = np.zeros((3 * sumElem, 3))
        # 按照单元进行循环
        for name in umat.nameList:
            D = umat.part[name].hessian()
            for elemID in umat.set[name]:
                elemB = self.evualB2D(x[elemID, :], y[elemID, :])
                # 组集B和D，计算应力
                Bmat[elemID * nge:(elemID + 1) * nge, :] = elemB
                Dmat[elemID * 3:(elemID + 1) * 3, :] = D
        return Bmat, Dmat

    def evualB2D(self, x, y):
        # Ke = np.zeros((self.mnode * 2, self.mnode * 2))
        elemB = np.zeros((len(self.hammer_int.points) * 3, self.mnode * 2))
        for i in range(len(self.hammer_int.points)):
            ks, yt = self.hammer_int.points[i]
            B = np.zeros((3, 2 * self.mnode))
            N_ks, N_yt = self.Dfun.dfun2D(ks, yt)
            NksNyt = np.concatenate((N_ks.T, N_yt.T))
            xy = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)))
            J = np.dot(NksNyt, xy.T)
            for n in range(1, self.mnode + 1):
                N_cor = np.linalg.solve(J, NksNyt[:, n - 1])
                B[0, 2 * n - 2] = N_cor[0]
                B[1, 2 * n - 1] = N_cor[1]
                B[2, 2 * n - 1] = N_cor[0]
                B[2, 2 * n - 2] = N_cor[1]
            elemB[3 * i:3 * (i + 1), :] = B
        return elemB
