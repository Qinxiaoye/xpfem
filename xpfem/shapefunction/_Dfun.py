import numpy as np


class Dfun:
    def __init__(self, mnode):
        self.mnode = mnode

    def dfun2D(self, ks, yt):
        N_ks = np.zeros((self.mnode, 1))
        N_yt = np.zeros((self.mnode, 1))
        if self.mnode == 4:
            N_ks[0] = (-1 + yt) / 4
            N_ks[1] = (1 - yt) / 4
            N_ks[2] = (1 + yt) / 4
            N_ks[3] = (-1 - yt) / 4
            N_yt[0] = (-1 + ks) / 4
            N_yt[1] = (-1 - ks) / 4
            N_yt[2] = (1 + ks) / 4
            N_yt[3] = (1 - ks) / 4
        else:
            N_ks[0] = (-1 / 4) * ((-1) + yt) * (2 * ks + yt)
            N_ks[1] = (1 / 4) * ((-1) + yt) * ((-2) * ks + yt)
            N_ks[2] = (1 / 4) * (1 + yt) * (2 * ks + yt)
            N_ks[3] = (-1 / 4) * (1 + yt) * ((-2) * ks + yt)
            N_ks[4] = ks * ((-1) + yt)
            N_ks[5] = (1 / 2) * (1 + (-1) * yt ** 2)
            N_ks[6] = (-1) * ks * (1 + yt)
            N_ks[7] = (1 / 2) * ((-1) + yt ** 2)
            N_yt[0] = (-1 / 4) * ((-1) + ks) * (ks + 2 * yt)
            N_yt[1] = (1 / 4) * (1 + ks) * ((-1) * ks + 2 * yt)
            N_yt[2] = (1 / 4) * (1 + ks) * (ks + 2 * yt)
            N_yt[3] = (-1 / 4) * ((-1) + ks) * ((-1) * ks + 2 * yt)
            N_yt[4] = (1 / 2) * ((-1) + ks ** 2)
            N_yt[5] = (-1) * (1 + ks) * yt
            N_yt[6] = (1 / 2) * (1 + (-1) * ks ** 2)
            N_yt[7] = ((-1) + ks) * yt
        return N_ks, N_yt

    def dfun3D(self, ks, yt, zita):
        N_ks = np.zeros((self.mnode, 1))
        N_yt = np.zeros((self.mnode, 1))
        N_zita = np.zeros((self.mnode, 1))
        if self.mnode == 8:
            N_ks[0] = (1 / 8) * ((-1) + yt) * ((-1) + zita)
            N_ks[1] = (-1 / 8) * (1 + yt) * ((-1) + zita)
            N_ks[2] = (1 / 8) * (1 + yt) * ((-1) + zita)
            N_ks[3] = (-1 / 8) * ((-1) + yt) * ((-1) + zita)
            N_ks[4] = (-1 / 8) * ((-1) + yt) * (1 + zita)
            N_ks[5] = (1 / 8) * (1 + yt) * (1 + zita)
            N_ks[6] = (-1 / 8) * (1 + yt) * (1 + zita)
            N_ks[7] = (1 / 8) * ((-1) + yt) * (1 + zita)

            N_yt[0] = (1 / 8) * (1 + ks) * ((-1) + zita)
            N_yt[1] = (-1 / 8) * (1 + ks) * ((-1) + zita)
            N_yt[2] = (1 / 8) * ((-1) + ks) * ((-1) + zita)
            N_yt[3] = (-1 / 8) * ((-1) + ks) * ((-1) + zita)
            N_yt[4] = (-1 / 8) * (1 + ks) * (1 + zita)
            N_yt[5] = (1 / 8) * (1 + ks) * (1 + zita)
            N_yt[6] = (-1 / 8) * ((-1) + ks) * (1 + zita)
            N_yt[7] = (1 / 8) * ((-1) + ks) * (1 + zita)

            N_zita[0] = (1 / 8) * (1 + ks) * ((-1) + yt)
            N_zita[1] = (-1 / 8) * (1 + ks) * (1 + yt)
            N_zita[2] = (1 / 8) * ((-1) + ks) * (1 + yt)
            N_zita[3] = (-1 / 8) * ((-1) + ks) * ((-1) + yt)
            N_zita[4] = (-1 / 8) * (1 + ks) * ((-1) + yt)
            N_zita[5] = (1 / 8) * (1 + ks) * (1 + yt)
            N_zita[6] = (-1 / 8) * ((-1) + ks) * (1 + yt)
            N_zita[7] = (1 / 8) * ((-1) + ks) * ((-1) + yt)
        else:
            N_ks[0] = (-1 / 8) * ((-1) + yt) * ((-1) + zita) * (1 + (-2) * ks + yt + zita)
            N_ks[1] = (1 / 8) * (1 + yt) * ((-1) + zita) * (1 + (-2) * ks + (-1) * yt + zita)
            N_ks[2] = (-1 / 8) * (1 + yt) * ((-1) + zita) * (1 + 2 * ks + (-1) * yt + zita)
            N_ks[3] = (1 / 8) * ((-1) + yt) * ((-1) + zita) * (1 + 2 * ks + yt + zita)
            N_ks[4] = (-1 / 8) * ((-1) + yt) * (1 + zita) * ((-1) + 2 * ks + (-1) * yt + zita)
            N_ks[5] = (1 / 8) * (1 + yt) * (1 + zita) * ((-1) + 2 * ks + yt + zita)
            N_ks[6] = (-1 / 8) * (1 + yt) * (1 + zita) * ((-1) + (-2) * ks + yt + zita)
            N_ks[7] = (1 / 8) * ((-1) + yt) * (1 + zita) * ((-1) + (-2) * ks + (-1) * yt + zita)
            N_ks[8] = (1 / 4) * ((-1) + yt ** 2) * ((-1) + zita)
            N_ks[9] = (1 / 2) * ks * (1 + yt) * ((-1) + zita)
            N_ks[10] = (-1 / 4) * ((-1) + yt ** 2) * ((-1) + zita)
            N_ks[11] = (-1 / 2) * ks * ((-1) + yt) * ((-1) + zita)
            N_ks[12] = (-1 / 4) * ((-1) + yt ** 2) * (1 + zita)
            N_ks[13] = (-1 / 2) * ks * (1 + yt) * (1 + zita)
            N_ks[14] = (1 / 4) * ((-1) + yt ** 2) * (1 + zita)
            N_ks[15] = (1 / 2) * ks * ((-1) + yt) * (1 + zita)
            N_ks[16] = (1 / 4) * ((-1) + yt) * ((-1) + zita ** 2)
            N_ks[17] = (-1 / 4) * (1 + yt) * ((-1) + zita ** 2)
            N_ks[18] = (1 / 4) * (1 + yt) * ((-1) + zita ** 2)
            N_ks[19] = (-1 / 4) * ((-1) + yt) * ((-1) + zita ** 2)

            N_yt[0] = (-1 / 8) * (1 + ks) * ((-1) + zita) * (1 + (-1) * ks + 2 * yt + zita)
            N_yt[1] = (1 / 8) * (1 + ks) * ((-1) + zita) * (1 + (-1) * ks + (-2) * yt + zita)
            N_yt[2] = (-1 / 8) * ((-1) + ks) * ((-1) + zita) * (1 + ks + (-2) * yt + zita)
            N_yt[3] = (1 / 8) * ((-1) + ks) * ((-1) + zita) * (1 + ks + 2 * yt + zita)
            N_yt[4] = (-1 / 8) * (1 + ks) * (1 + zita) * ((-1) + ks + (-2) * yt + zita)
            N_yt[5] = (1 / 8) * (1 + ks) * (1 + zita) * ((-1) + ks + 2 * yt + zita)
            N_yt[6] = (-1 / 8) * ((-1) + ks) * (1 + zita) * ((-1) + (-1) * ks + 2 * yt + zita)
            N_yt[7] = (1 / 8) * ((-1) + ks) * (1 + zita) * ((-1) + (-1) * ks + (-2) * yt + zita)
            N_yt[8] = (1 / 2) * (1 + ks) * yt * ((-1) + zita)
            N_yt[9] = (1 / 4) * ((-1) + ks ** 2) * ((-1) + zita)
            N_yt[10] = (-1 / 2) * ((-1) + ks) * yt * ((-1) + zita)
            N_yt[11] = (-1 / 4) * ((-1) + ks ** 2) * ((-1) + zita)
            N_yt[12] = (-1 / 2) * (1 + ks) * yt * (1 + zita)
            N_yt[13] = (-1 / 4) * ((-1) + ks ** 2) * (1 + zita)
            N_yt[14] = (1 / 2) * ((-1) + ks) * yt * (1 + zita)
            N_yt[15] = (1 / 4) * ((-1) + ks ** 2) * (1 + zita)
            N_yt[16] = (1 / 4) * (1 + ks) * ((-1) + zita ** 2)
            N_yt[17] = (-1 / 4) * (1 + ks) * ((-1) + zita ** 2)
            N_yt[18] = (1 / 4) * ((-1) + ks) * ((-1) + zita ** 2)
            N_yt[19] = (-1 / 4) * ((-1) + ks) * ((-1) + zita ** 2)

            N_zita[0] = (-1 / 8) * (1 + ks) * ((-1) + yt) * (1 + (-1) * ks + yt + 2 * zita)
            N_zita[1] = (-1 / 8) * (1 + ks) * (1 + yt) * ((-1) + ks + yt + (-2) * zita)
            N_zita[2] = (1 / 8) * ((-1) + ks) * (1 + yt) * ((-1) + (-1) * ks + yt + (-2) * zita)
            N_zita[3] = (1 / 8) * ((-1) + ks) * ((-1) + yt) * (1 + ks + yt + 2 * zita)
            N_zita[4] = (1 / 8) * (1 + ks) * ((-1) + yt) * (1 + (-1) * ks + yt + (-2) * zita)
            N_zita[5] = (1 / 8) * (1 + ks) * (1 + yt) * ((-1) + ks + yt + 2 * zita)
            N_zita[6] = (-1 / 8) * ((-1) + ks) * (1 + yt) * ((-1) + (-1) * ks + yt + 2 * zita)
            N_zita[7] = (-1 / 8) * ((-1) + ks) * ((-1) + yt) * (1 + ks + yt + (-2) * zita)
            N_zita[8] = (1 / 4) * (1 + ks) * ((-1) + yt ** 2)
            N_zita[9] = (1 / 4) * ((-1) + ks ** 2) * (1 + yt)
            N_zita[10] = (-1 / 4) * ((-1) + ks) * ((-1) + yt ** 2)
            N_zita[11] = (-1 / 4) * ((-1) + ks ** 2) * ((-1) + yt)
            N_zita[12] = (-1 / 4) * (1 + ks) * ((-1) + yt ** 2)
            N_zita[13] = (-1 / 4) * ((-1) + ks ** 2) * (1 + yt)
            N_zita[14] = (1 / 4) * ((-1) + ks) * ((-1) + yt ** 2)
            N_zita[15] = (1 / 4) * ((-1) + ks ** 2) * ((-1) + yt)
            N_zita[16] = (1 / 2) * (1 + ks) * ((-1) + yt) * zita
            N_zita[17] = (-1 / 2) * (1 + ks) * (1 + yt) * zita
            N_zita[18] = (1 / 2) * ((-1) + ks) * (1 + yt) * zita
            N_zita[19] = (-1 / 2) * ((-1) + ks) * ((-1) + yt) * zita
        return N_ks, N_yt, N_zita

