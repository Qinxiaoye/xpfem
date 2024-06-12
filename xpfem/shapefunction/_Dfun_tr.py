import numpy as np


class Dfun_tr:
    def __init__(self, mnode):
        self.mnode = mnode

    def dfun2D(self, ks, yt):
        N_ks = np.zeros((self.mnode, 1))
        N_yt = np.zeros((self.mnode, 1))
        N_zita = np.zeros((self.mnode, 1))
        if self.mnode == 3:
            N_ks[0] = 1
            N_yt[1] = 1
            N_ks[2] = -1
            N_yt[2] = -1

        else:
            N_ks[0] = 4 * ks - 1
            N_ks[2] = 4 * (ks + yt) - 3
            N_ks[3] = 4 * yt
            N_ks[4] = -4 * yt
            N_ks[5] = 4 - 4 * yt - 8 * ks

            N_yt[1] = 4 * yt - 1
            N_yt[2] = 4 * (ks + yt) - 3
            N_yt[3] = 4 * ks
            N_yt[4] = 4 - 4 * ks - 8 * yt
            N_yt[5] = -4 * ks

        return N_ks, N_yt

    def dfun3D(self, ks, yt, zita):
        N_ks = np.zeros((self.mnode, 1))
        N_yt = np.zeros((self.mnode, 1))
        N_zita = np.zeros((self.mnode, 1))
        if self.mnode == 4:
            N_ks[0] = 1
            N_yt[1] = 1
            N_zita[2] = 1
            N_ks[3] = -1
            N_yt[3] = -1
            N_zita[3] = -1
        else:  #
            N_ks[0] = 4 * ks - 1
            N_yt[1] = 4 * yt - 1
            N_zita[2] = 4 * zita - 1
            N_ks[3] = 4 * (ks + yt + zita) - 3
            N_yt[3] = 4 * (ks + yt + zita) - 3
            N_zita[3] = 4 * (ks + yt + zita) - 3
            N_ks[4] = 4 * yt
            N_yt[4] = 4 * ks
            N_ks[5] = 4 * zita
            N_zita[5] = 4 * ks
            N_ks[6] = 4 * (1 - yt - zita - 2 * ks)
            N_yt[6] = -4 * ks
            N_zita[6] = -4 * ks
            N_yt[7] = 4 * zita
            N_zita[7] = 4 * yt
            N_ks[8] = -4 * zita
            N_yt[8] = -4 * zita
            N_zita[8] = 4 * (1 - ks - yt - 2 * zita)
            N_ks[9] = -4 * yt
            N_yt[9] = 4 * (1 - ks - 2 * yt - zita)
            N_zita[9] = -4 * yt
        return N_ks, N_yt, N_zita
