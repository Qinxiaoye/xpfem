import numpy as np


class GlobalK3D_C3D4:
    def __init__(self):
        self.mnode = 4

    def GK(self, elem, node, EX, MU):
        sumElem = elem.shape[0]
        u = np.array([])
        v = np.array([])
        a = np.array([])
        x = np.zeros((sumElem, self.mnode))
        y = np.zeros((sumElem, self.mnode))
        z = np.zeros((sumElem, self.mnode))
        for m in range(1, self.mnode + 1):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
            z[:, m - 1] = node[elem[:, m - 1], 2]
        for n in range(1, sumElem + 1):
            Ex = EX[n - 1]
            Mu = MU[n - 1]
            K = self.elemK3D(x[n - 1, :], y[n - 1, :], z[n - 1, :], Ex, Mu)
            nodeID = elem[n - 1, :]
            ndfID = np.zeros(3 * self.mnode)
            ndfID[0::3] = nodeID * 3
            ndfID[1::3] = nodeID * 3 + 1
            ndfID[2::3] = nodeID * 3 + 2
            ddd = np.tile(ndfID, (3 * self.mnode, 1))
            u = np.concatenate((u, ddd.flatten('F')))
            v = np.concatenate((v, np.tile(ndfID, 3 * self.mnode)))
            a = np.concatenate((a, K.flatten('F')))
        return u, v, a

    def elemK3D(self, x, y, z, Ex, mu):
        Ke = np.zeros((self.mnode * 2, self.mnode * 2))
        D = Ex * (1 - mu) / ((1 + mu) * (1 - 2 * mu)) * \
            np.array([[1, mu / (1 - mu), mu / (1 - mu), 0, 0, 0],
                      [mu / (1 - mu), 1, mu / (1 - mu), 0, 0, 0],
                      [mu / (1 - mu), mu / (1 - mu), 1, 0, 0, 0],
                      [0, 0, 0, (1 - 2 * mu) / (2 * (1 - mu)), 0, 0],
                      [0, 0, 0, 0, (1 - 2 * mu) / (2 * (1 - mu)), 0],
                      [0, 0, 0, 0, 0, (1 - 2 * mu) / (2 * (1 - mu))]])

        B, V = elemB_linear(x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1))
        Ke = np.dot(np.dot(B.T, D), B) * V
        return Ke


def elemB_linear(x, y, z):
    B = np.zeros((6, 12))
    # bb = np.ones((3, 1))
    # a = np.concatenate((x, y, z))  # alpha
    b = np.concatenate((np.ones((1, 4)), y, z))  # beta
    c = np.concatenate((np.ones((1, 4)), x, z))  # gama
    d = np.concatenate((np.ones((1, 4)), x, y))  # lamda
    Vm = np.concatenate((np.ones((1, 4)), x, y, z))
    V6 = np.linalg.det(Vm)
    for i in range(1, 5):
        # a2 = -1 ** (1+i) * np.linalg.det(np.delete(a, i - 1, 1))
        b1 = np.delete(b, i - 1, 1)
        c1 = np.delete(c, i - 1, 1)
        d1 = np.delete(d, i - 1, 1)
        b2 = (-1) ** i * np.linalg.det(b1)
        c2 = (-1) ** (1 + i) * np.linalg.det(c1)
        d2 = (-1) ** i * np.linalg.det(d1)
        B[0, 3 * i - 3] = b2
        B[1, 3 * i - 2] = c2
        B[2, 3 * i - 1] = d2
        B[3, 3 * i - 3] = c2
        B[3, 3 * i - 2] = b2
        B[4, 3 * i - 2] = d2
        B[4, 3 * i - 1] = c2
        B[5, 3 * i - 3] = d2
        B[5, 3 * i - 1] = b2
    B = B / V6
    return B, V6 / 6
