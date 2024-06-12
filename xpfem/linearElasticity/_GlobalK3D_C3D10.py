import numpy as np


class GlobalK3D_C3D10:
    def __init__(self, Dfun):
        self.Dfun = Dfun

    def GK(self, elem, node, EX, MU):
        sumElem = elem.shape[0]
        u = np.array([])
        v = np.array([])
        a = np.array([])
        x = np.zeros((sumElem, 10))
        y = np.zeros((sumElem, 10))
        z = np.zeros((sumElem, 10))
        for m in range(1, 11):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
            z[:, m - 1] = node[elem[:, m - 1], 2]
        for n in range(1, sumElem + 1):
            Ex = EX[n - 1]
            Mu = MU[n - 1]
            K = self.elemK3D(x[n - 1, :], y[n - 1, :], z[n - 1, :], Ex, Mu)
            nodeID = elem[n - 1, :]
            ndfID = np.zeros(3 * 10)
            ndfID[0::3] = nodeID * 3
            ndfID[1::3] = nodeID * 3 + 1
            ndfID[2::3] = nodeID * 3 + 2
            ddd = np.tile(ndfID, (3 * 10, 1))
            u = np.concatenate((u, ddd.flatten('F')))
            v = np.concatenate((v, np.tile(ndfID, 3 * 10)))
            a = np.concatenate((a, K.flatten('F')))
        return u, v, a

    def elemK3D(self, x, y, z, Ex, mu):
        Ke = np.zeros((30, 30))
        D = Ex * (1 - mu) / ((1 + mu) * (1 - 2 * mu)) * \
            np.array([[1, mu / (1 - mu), mu / (1 - mu), 0, 0, 0],
                      [mu / (1 - mu), 1, mu / (1 - mu), 0, 0, 0],
                      [mu / (1 - mu), mu / (1 - mu), 1, 0, 0, 0],
                      [0, 0, 0, (1 - 2 * mu) / (2 * (1 - mu)), 0, 0],
                      [0, 0, 0, 0, (1 - 2 * mu) / (2 * (1 - mu)), 0],
                      [0, 0, 0, 0, 0, (1 - 2 * mu) / (2 * (1 - mu))]])
        nip = 4
        # ks = np.array([[1 / 4, 1 / 4, 1 / 4],
        #                [1 / 2, 1 / 6, 1 / 6],
        #                [1 / 6, 1 / 2, 1 / 6],
        #                [1 / 6, 1 / 6, 1 / 2],
        #                [1 / 6, 1 / 6, 1 / 6]])
        # w = np.array([-2 / 15, 3 / 40, 3 / 40, 3 / 40, 3 / 40])
        ks = np.array([
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
            [0.13819660, 0.13819660, 0.13819660]])
        w = np.array([1 / 24, 1 / 24, 1 / 24, 1 / 24])
        for n in range(1, nip + 1):
            J, B = self.elemB3D(x, y, z, ks[n - 1, 0], ks[n - 1, 1], ks[n - 1, 2])
            Ke = Ke + w[n - 1] * np.dot(np.dot(B.T, D), B) * np.linalg.det(J)
        return Ke

    def elemB3D(self, x, y, z, ks, yt, zita):
        B = np.zeros((6, 3 * 10))
        N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
        NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
        xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
        J = np.dot(NksNytNzita, xyz.T)
        for n in range(1, 11):
            N_cor = np.linalg.solve(J, NksNytNzita[:, n - 1])
            B[0, 3 * n - 3] = N_cor[0]
            B[1, 3 * n - 2] = N_cor[1]
            B[2, 3 * n - 1] = N_cor[2]
            B[3, 3 * n - 3] = N_cor[1]
            B[3, 3 * n - 2] = N_cor[0]
            B[4, 3 * n - 2] = N_cor[2]
            B[4, 3 * n - 1] = N_cor[1]
            B[5, 3 * n - 3] = N_cor[2]
            B[5, 3 * n - 1] = N_cor[0]
        return J, B
