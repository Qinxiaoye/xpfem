import numpy as np


class GlobalK2D_C2D6:
    def __init__(self, Dfun, isStress):
        self.Dfun = Dfun
        self.isStress = isStress

    def GK(self, elem, node, EX, MU):
        sumElem = elem.shape[0]
        u = np.array([])
        v = np.array([])
        a = np.array([])
        x = np.zeros((sumElem, 6))
        y = np.zeros((sumElem, 6))
        for m in range(1, 6 + 1):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
        for n in range(1, sumElem + 1):
            Ex = EX[n - 1]
            Mu = MU[n - 1]
            K = self.elemK2D(x[n - 1, :], y[n - 1, :], Ex, Mu)
            nodeID = elem[n - 1, :]
            ndfID = np.zeros(2 * 6)
            ndfID[0::2] = nodeID * 2
            ndfID[1::2] = nodeID * 2 + 1
            ddd = np.tile(ndfID, (2 * 6, 1))
            u = np.concatenate((u, ddd.flatten('F')))
            v = np.concatenate((v, np.tile(ndfID, 2 * 6)))
            a = np.concatenate((a, K.flatten('F')))
        return u, v, a

    def elemK2D(self, x, y, Ex, mu):
        Ke = np.zeros((6 * 2, 6 * 2))
        if self.isStress == 1:
            # 平面应力
            D = Ex / (1 - mu ** 2) * np.array([[1, mu, 0], [mu, 1, 0], [0, 0, (1 - mu) / 2]])
        else:
            # 平面应变
            D = Ex * (1 - mu) / ((1 + mu) * (1 - 2 * mu)) * np.array(
                [[1, mu / (1 - mu), 0], [mu / (1 - mu), 1, 0], [0, 0, (1 - 2 * mu) / (2 * (1 - mu))]])
        nip = 3
        ks = np.array([[1/6, 2 / 3],
                       [2 / 3, 1 / 6], [1 / 6, 1 / 6]])
        w = [1 / 3, 1 / 3, 1 / 3]
        for n in range(1, nip + 1):
            J, B = self.elemB2D(x, y, ks[n - 1, 0], ks[n - 1, 1])
            Ke = Ke + w[n - 1] * np.dot(np.dot(B.T, D), B) * np.linalg.det(J)
        return Ke

    def elemB2D(self, x, y, ks, yt):
        B = np.zeros((3, 2 * 6))
        N_ks, N_yt = self.Dfun.dfun2D(ks, yt)
        NksNyt = np.concatenate((N_ks.T, N_yt.T))
        xy = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)))
        J = np.dot(NksNyt, xy.T)
        for n in range(1, 6 + 1):
            N_cor = np.linalg.solve(J, NksNyt[:, n - 1])
            B[0, 2 * n - 2] = N_cor[0]
            B[1, 2 * n - 1] = N_cor[1]
            B[2, 2 * n - 1] = N_cor[0]
            B[2, 2 * n - 2] = N_cor[1]
        return J, B
