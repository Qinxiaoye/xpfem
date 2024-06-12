import numpy as np


class Shapefunc_tr:
    def __init__(self, mnode, ndim):
        self.mnode = mnode
        self.ndim = ndim

    def fun(self, ks, yt, zita):
        N = np.zeros(self.mnode)
        if self.ndim == 2:
            if self.mnode == 3:
                N[0] = ks
                N[1] = yt
                N[2] = 1 - ks - yt
            else:
                N[0] = ks * (2 * ks - 1)
                N[1] = yt * (2 * yt - 1)
                N[2] = (1 - ks - yt) * (2 * (1 - ks - yt) - 1)
                N[3] = 4 * ks * yt
                N[4] = 4 * yt * (1 - ks - yt)
                N[5] = 4 * (1 - ks - yt) * ks

        else:
            if self.mnode == 4:
                N[0] = ks
                N[1] = yt
                N[2] = zita
                N[3] = 1 - ks - yt - zita
            else:
                N[0] = (2 * ks - 1) * ks
                N[1] = (2 * yt - 1) * yt
                N[2] = (2 * zita - 1) * zita
                N[3] = (2 * (1 - ks - yt - zita) - 1) * (1 - ks - yt - zita)
                N[4] = 4 * ks * yt
                N[5] = 4 * ks * zita
                N[6] = 4 * ks * (1 - ks - yt - zita)
                N[7] = 4 * yt * zita
                N[8] = 4 * zita * (1 - ks - yt - zita)
                N[9] = 4 * yt * (1 - ks - yt - zita)

        return N
