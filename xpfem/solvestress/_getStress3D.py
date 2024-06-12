import numpy as np
from xpfem import shapefunction, math


class getStress3D:
    def __init__(self, mnode):
        self.Dfun = shapefunction.Dfun(mnode)
        self.mnode = mnode

    def evualStress(self, displacement, region):
        elem = region.mesh.cells
        node = region.mesh.points
        displacement = displacement.reshape(-1, 3)
        sumElem = elem.shape[0]
        sumNode = node.shape[0]
        Bmat, Dmat = self.evualMassB(region)
        s = np.zeros((sumElem * 48, 1))
        stress = np.zeros((sumNode, 6))
        nodeMes = np.zeros((sumNode, 1))
        a = (5 + 3 * np.sqrt(3)) / 4
        b = -(np.sqrt(3) + 1) / 4
        c = (np.sqrt(3) - 1) / 4
        d = (5 - 3 * np.sqrt(3)) / 4
        invs = np.array([
            [-0.683012701892218,  0.183012701892219,  0.183012701892219, -0.049038105676658,  2.549038105676656, -0.683012701892218, -0.683012701892218,  0.183012701892219],
            [ 0.183012701892219, -0.049038105676658, -0.683012701892218,  0.183012701892219, -0.683012701892218,  0.183012701892219,  2.549038105676656, -0.683012701892218],
            [-0.683012701892218,  0.183012701892219,  2.549038105676656, -0.683012701892218,  0.183012701892219, -0.049038105676658, -0.683012701892218,  0.183012701892219],
            [ 2.549038105676656, -0.683012701892218, -0.683012701892218,  0.183012701892219, -0.683012701892218,  0.183012701892219,  0.183012701892219, -0.049038105676658],
            [ 0.183012701892219, -0.683012701892218, -0.049038105676658,  0.183012701892219, -0.683012701892218,  2.549038105676656,  0.183012701892219, -0.683012701892218],
            [-0.049038105676658,  0.183012701892219,  0.183012701892219, -0.683012701892218,  0.183012701892219, -0.683012701892218, -0.683012701892218,  2.549038105676656],
            [ 0.183012701892219, -0.683012701892218, -0.683012701892218,  2.549038105676656, -0.049038105676658,  0.183012701892219,  0.183012701892219, -0.683012701892218],
            [-0.683012701892218,  2.549038105676656,  0.183012701892219, -0.683012701892218,  0.183012701892219, -0.683012701892218, -0.049038105676658,  0.183012701892219]
        ])

        dbu = np.zeros((48, 1))
        for n in range(sumElem):
            elemU = displacement[elem[n, :], :]
            elemU = elemU.reshape(-1, 1)
            Bu = np.dot(Bmat[48 * n:48 * (n + 1), :], elemU)
            for m in range(8):
                dbu[6 * m:6 * (m + 1)] = np.dot(Dmat[n * 6:6 * (n + 1), :], Bu[m * 6:6 * (m + 1)])
            s[48 * n: 48 * (n + 1)] = dbu
        s = s.reshape(-1, 6)
        for n in range(sumElem):
            nodeMes[elem[n, :]] += 1  # 节点被几个单元共用
            stress[elem[n, 0:8], :] += np.dot(invs,s[8 * n:8 * (n + 1), :])
        stress = stress / nodeMes
        if self.mnode == 20:
            stress[elem[:, 8], :] = (stress[elem[:, 0], :] + stress[elem[:, 1], :]) / 2
            stress[elem[:, 9], :] = (stress[elem[:, 1], :] + stress[elem[:, 2], :]) / 2
            stress[elem[:, 10], :] = (stress[elem[:, 2], :] + stress[elem[:, 3], :]) / 2
            stress[elem[:, 11], :] = (stress[elem[:, 3], :] + stress[elem[:, 0], :]) / 2
            stress[elem[:, 12], :] = (stress[elem[:, 4], :] + stress[elem[:, 5], :]) / 2
            stress[elem[:, 13], :] = (stress[elem[:, 5], :] + stress[elem[:, 6], :]) / 2
            stress[elem[:, 14], :] = (stress[elem[:, 6], :] + stress[elem[:, 7], :]) / 2
            stress[elem[:, 15], :] = (stress[elem[:, 7], :] + stress[elem[:, 4], :]) / 2
            stress[elem[:, 16], :] = (stress[elem[:, 0], :] + stress[elem[:, 4], :]) / 2
            stress[elem[:, 17], :] = (stress[elem[:, 1], :] + stress[elem[:, 5], :]) / 2
            stress[elem[:, 18], :] = (stress[elem[:, 2], :] + stress[elem[:, 6], :]) / 2
            stress[elem[:, 19], :] = (stress[elem[:, 3], :] + stress[elem[:, 7], :]) / 2
        a = ((stress[:, 0] - stress[:, 1])**2+
             (stress[:, 1] - stress[:, 2])**2+
             (stress[:, 0] - stress[:, 2])**2)/2
        b = (stress[:, 3] ** 2+stress[:, 4] ** 2+stress[:, 5] ** 2)*3
        mises = np.sqrt(a+b)
        return stress,mises

    def evualMassB(self, region):
        elem = region.mesh.cells
        node = region.mesh.points
        umat = region.umat
        sumElem = elem.shape[0]
        nnz0 = (self.mnode * 3) * (self.mnode * 3)
        x = np.zeros((sumElem, self.mnode))
        y = np.zeros((sumElem, self.mnode))
        z = np.zeros((sumElem, self.mnode))
        for m in range(1, self.mnode + 1):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
            z[:, m - 1] = node[elem[:, m - 1], 2]

        # 高斯积分,三维维的固定为2阶，理论上八节点可以使用一阶
        self.gauss_int = math.gauss_legendre(intOrder=2, dim=3)
        nge = len(self.gauss_int.points) * 6
        Bmat = np.zeros((nge * sumElem, self.mnode * 3))
        Dmat = np.zeros((6 * sumElem, 6))
        # 按照单元进行循环
        for name in umat.nameList:
            D = umat.part[name].hessian()
            for elemID in umat.set[name]:
                elemB = self.evualB3D(x[elemID, :], y[elemID, :], z[elemID, :])
                # 组集B和D，计算应力
                Bmat[elemID * nge:(elemID + 1) * nge, :] = elemB
                Dmat[elemID * 6:(elemID + 1) * 6, :] = D
        return Bmat, Dmat

    def evualB3D(self, x, y, z):
        # Ke = np.zeros((self.mnode * 2, self.mnode * 2))
        elemB = np.zeros((len(self.gauss_int.points) * 6, self.mnode * 3))
        for i in range(len(self.gauss_int.points)):
            ks, yt, zita = self.gauss_int.points[i]
            B = np.zeros((6, 3 * self.mnode))
            N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
            NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
            xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
            J = np.dot(NksNytNzita, xyz.T)
            N_cor = np.linalg.solve(J, NksNytNzita)
            for n in range(1, self.mnode + 1):
                B[0, 3 * n - 3] = N_cor[0, n - 1]
                B[1, 3 * n - 2] = N_cor[1, n - 1]
                B[2, 3 * n - 1] = N_cor[2, n - 1]
                B[3, 3 * n - 3] = N_cor[1, n - 1]
                B[3, 3 * n - 2] = N_cor[0, n - 1]
                B[4, 3 * n - 2] = N_cor[2, n - 1]
                B[4, 3 * n - 1] = N_cor[1, n - 1]
                B[5, 3 * n - 3] = N_cor[2, n - 1]
                B[5, 3 * n - 1] = N_cor[0, n - 1]
            elemB[6 * i:6 * (i + 1), :] = B
        return elemB
