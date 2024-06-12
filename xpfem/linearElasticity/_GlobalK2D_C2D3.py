import numpy as np


class GlobalK2D_C2D3:
    def __init__(self, isStress):
        self.isStress = isStress

    def GK(self, elem, node, EX, MU):
        sumElem = elem.shape[0]
        u = np.array([])
        v = np.array([])
        a = np.array([])
        x = np.zeros((sumElem, 3))
        y = np.zeros((sumElem, 3))
        for m in range(1, 4):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
        for n in range(1, sumElem + 1):
            Ex = EX[n - 1]
            Mu = MU[n - 1]
            K = self.elemK2D(x[n - 1, :], y[n - 1, :], Ex, Mu)
            nodeID = elem[n - 1, :]
            ndfID = np.zeros(6)
            ndfID[0::2] = nodeID * 2
            ndfID[1::2] = nodeID * 2 + 1
            ddd = np.tile(ndfID, (6, 1))
            u = np.concatenate((u, ddd.flatten('F')))
            v = np.concatenate((v, np.tile(ndfID, 6)))
            a = np.concatenate((a, K.flatten('F')))
        return u, v, a

    def elemK2D(self, x, y, Ex, mu):
        Ke = np.zeros((6, 6))
        if self.isStress == 1:
            # 平面应力
            D = Ex / (1 - mu ** 2) * np.array([[1, mu, 0], [mu, 1, 0], [0, 0, (1 - mu) / 2]])
        else:
            # 平面应变
            D = Ex * (1 - mu) / ((1 + mu) * (1 - 2 * mu)) * np.array(
                [[1, mu / (1 - mu), 0], [mu / (1 - mu), 1, 0], [0, 0, (1 - 2 * mu) / (2 * (1 - mu))]])

        A, B = elemB_linear(x, y)
        Ke = np.dot(np.dot(B.T, D), B) * A
        return Ke


def elemB_linear(x, y):
    XX = np.array([[2, 3], [3, 1], [1, 2]])
    B = np.zeros((3, 6))
    A = 1 / 2 * (x[2] * y[3] - x[3] * y[2] + x[3] * y[1] - x[1] * y[3] + x[1] * y[2] - x[2] * y[1])
    for i in range(1, 4):
        b = y[XX[i, 0]] - y[XX[i, 1]]
        c = x[XX[i, 1]] - x[XX[i, 0]]
        B[0, 2 * i - 2] = b
        B[1, 2 * i - 1] = c
        B[2, 2 * i - 1] = b
        B[2, 2 * i - 2] = c
    B = B / (2 * A)
    return A, B
