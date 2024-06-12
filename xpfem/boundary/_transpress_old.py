import numpy as np
from xpfem import shapefunction

class transPress:

    def __init__(self, mnode, ndim,press):
        self.Dfun = shapefunction.Dfun(mnode)
        self.Shapefunc = shapefunction.Shapefunc(mnode, ndim)
        self.mnode = mnode
        self.ndim = ndim
        self.press = press

    def transpress(self, node, elem):
        sumPre = self.press.shape[0]
        if self.ndim == 2:
            if self.mnode == 4:
                nfsize = 4
            else:
                nfsize = 6
        else:
            if self.mnode == 8:
                nfsize = 12
            else:
                nfsize = 24
        nodeForceNew = np.zeros((nfsize * sumPre, 3))
        elemP = self.press[:, 0]
        face = self.press[:, 1]
        pres = self.press[:, 2]
        elemP = elemP.astype('int')
        face = face.astype('int')
        elemP = elemP - 1
        face = face - 1
        if self.ndim == 2:
            
            nodeForceNew[0::2, 1] = 1
            nodeForceNew[1::2, 1] = 2
            if self.mnode == 4:
                faceNode = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
                for i in range(1, sumPre + 1):
                    bNode = elem[elemP[i - 1], faceNode[face[i - 1], :]]
                    nodeForceNew[nfsize * (i - 1):nfsize * (i - 1) + 2, 0] = bNode[0]
                    nodeForceNew[nfsize * (i - 1) + 2:nfsize * i, 0] = bNode[1]
                    #
                    nodeForceNew[nfsize * (i - 1), 2] = 0.5 * pres[i - 1] * (
                            node[bNode[0], 1] - node[bNode[1], 1])
                    nodeForceNew[nfsize * (i - 1) + 1, 2] = 0.5 * pres[i - 1] * (
                            node[bNode[1], 0] - node[bNode[0], 0])
                    nodeForceNew[nfsize * (i - 1) + 2, 2] = 0.5 * pres[i - 1] * (
                            node[bNode[0], 1] - node[bNode[1], 1])
                    nodeForceNew[nfsize * (i - 1) + 3, 2] = 0.5 * pres[i - 1] * (
                            node[bNode[1], 0] - node[bNode[0], 0])
            else:
                faceNode = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]], dtype=int)
                for i in range(1, sumPre + 1):
                    bNode = elem[elemP[i - 1], faceNode[face[i - 1], :]]
                    nodeForceNew[nfsize * (i - 1):nfsize * (i - 1) + 2, 0] = bNode[0],
                    nodeForceNew[nfsize * (i - 1) + 2:nfsize * (i - 1) + 4, 0] = bNode[1]
                    nodeForceNew[nfsize * (i - 1) + 4:nfsize * i, 0] = bNode[2]
                    #
                    nodeForceNew[nfsize * (i - 1), 2] = pres[i - 1] * (
                            1 / 2 * node[bNode[0], 1] + 1 / 6 * node[bNode[1], 1] - 2 / 3 * node[bNode[2], 1])
                    nodeForceNew[nfsize * (i - 1) + 1, 2] = -1 * pres[i - 1] * (
                            1 / 2 * node[bNode[0], 0] + 1 / 6 * node[bNode[1], 0] - 2 / 3 * node[bNode[2], 0])
                    nodeForceNew[nfsize * (i - 1) + 2, 2] = -1 * pres[i - 1] * (
                            1 / 6 * node[bNode[0], 1] + 1 / 2 * node[bNode[1], 1] - 2 / 3 * node[bNode[2], 1])
                    nodeForceNew[nfsize * (i - 1) + 3, 2] = pres[i - 1] * (
                            1 / 6 * node[bNode[0], 0] + 1 / 2 * node[bNode[1], 0] - 2 / 3 * node[bNode[2], 0])
                    nodeForceNew[nfsize * (i - 1) + 4, 2] = 2 / 3 * pres[i - 1] * (
                            node[bNode[0], 1] - node[bNode[1], 1])
                    nodeForceNew[nfsize * (i - 1) + 5, 2] = -2 / 3 * pres[i - 1] * (
                            node[bNode[0], 0] - node[bNode[1], 0])
        else:
            nodeForceNew[0::3, 1] = 1
            nodeForceNew[1::3, 1] = 2
            nodeForceNew[2::3, 1] = 3
            nip = 3
            ks = np.array([-0.774596669241483, 0, 0.774596669241483])
            w = np.array([0.55555555555556, 0.888888888888888889, 0.55555555555556])
            if self.mnode == 8:
                faceNode = np.array([[1,4,3,2], [1,2,6,5], [2,3,7,6], [3,4,8,7], [1,5,8,4], [5,6,7,8]]).astype('int')-1
            else:
                faceNode = np.array([[1,4,3,2,9,12,11,10], [1,2,6,5,9,18,13,17],
                                     [2, 3, 7, 6, 10, 19, 14, 18], [3, 4, 8, 7, 11, 20, 15, 19],
                                     [1, 5, 8, 4, 17, 16, 20, 12], [5, 6, 7, 8, 13, 14, 15, 16]]).astype('int')-1
            for n in range(1, sumPre+1):
                bNode = elem[elemP[n - 1], faceNode[face[n - 1], :]]
                x = node[elem[elemP[n - 1], :], 0]
                y = node[elem[elemP[n - 1], :], 1]
                z = node[elem[elemP[n - 1], :], 2]
                re = np.tile(bNode, (3, 1))
                re = re.flatten('F')
                nodeForceNew[nfsize * (n - 1):nfsize * n, 0] = re[:]
                if face[n - 1] == 0:
                    for m in range(1, faceNode.shape[1]+1):
                        p = np.array([0, 0, 0])
                        for i in range(1, nip+1):
                            for j in range(1, nip+1):
                                fx, fy, fz = self.npc(x, y, z, ks[i - 1], ks[j - 1], -1, faceNode[0, m - 1])
                                p = p + w[i-1] * w[j-1] * np.array([fx, fy, fz])
                        nodeForceNew[nfsize * (n - 1) + 3 * m - 3: nfsize * (n - 1) + 3 * m, 2] = p * pres[n-1]
                elif face[n - 1] == 1:
                    for m in range(1, faceNode.shape[1]+1):
                        p = np.array([0, 0, 0])
                        for i in range(1, nip+1):
                            for j in range(1, nip+1):
                                fx, fy, fz = self.npc(x, y, z, 1, ks[i - 1], ks[j - 1], faceNode[1, m - 1])
                                p = p + w[i-1] * w[j-1] * np.array([fx, fy, fz])
                        nodeForceNew[nfsize * (n - 1) + 3 * m - 3: nfsize * (n - 1) + 3 * m, 2] = -p * pres[n-1]
                elif face[n - 1] == 2:
                    for m in range(1, faceNode.shape[1]+1):
                        p = np.array([0, 0, 0])
                        for i in range(1, nip+1):
                            for j in range(1, nip+1):
                                fx, fy, fz = self.npc(x, y, z, ks[i - 1], 1, ks[j - 1], faceNode[2, m - 1])
                                p = p + w[i-1] * w[j-1] * np.array([fx, fy, fz])
                        nodeForceNew[nfsize * (n - 1) + 3 * m - 3: nfsize * (n - 1) + 3 * m, 2] = -p * pres[n-1]
                elif face[n - 1] == 3:
                    for m in range(1, faceNode.shape[1]+1):
                        p = np.array([0, 0, 0])
                        for i in range(1, nip+1):
                            for j in range(1, nip+1):
                                fx, fy, fz = self.npc(x, y, z, -1, ks[i - 1], ks[j - 1], faceNode[3, m - 1])
                                p = p + w[i-1] * w[j-1] * np.array([fx, fy, fz])
                        nodeForceNew[nfsize * (n - 1) + 3 * m - 3: nfsize * (n - 1) + 3 * m, 2] = p * pres[n-1]
                elif face[n - 1] == 4:
                    for m in range(1, faceNode.shape[1]+1):
                        p = np.array([0, 0, 0])
                        for i in range(1, nip+1):
                            for j in range(1, nip+1):
                                fx, fy, fz = self.npc(x, y, z, ks[i - 1], -1, ks[j - 1], faceNode[4, m - 1])
                                p = p + w[i-1] * w[j-1] * np.array([fx, fy, fz])
                        nodeForceNew[nfsize * (n - 1) + 3 * m - 3: nfsize * (n - 1) + 3 * m, 2] = p * pres[n-1]
                elif face[n - 1] == 5:
                    for m in range(1, faceNode.shape[1]+1):
                        p = np.array([0, 0, 0])
                        for i in range(1, nip+1):
                            for j in range(1, nip+1):
                                fx, fy, fz = self.npc(x, y, z, ks[i - 1], ks[j - 1], 1, faceNode[5, m - 1])
                                p = p + w[i-1] * w[j-1] * np.array([fx, fy, fz])
                        nodeForceNew[nfsize * (n - 1) + 3 * m - 3: nfsize * (n - 1) + 3 * m, 2] = p * pres[n-1]
        sumNode = node.shape[0]
        nodeForce = np.zeros(sumNode*self.ndim)
        if sumPre > 0:
            nodeForce[(nodeForceNew[:, 0]) * self.ndim + nodeForceNew[:, 1] - 1] = nodeForceNew[:, 2]
        return nodeForce

    def npc(self, x, y, z, ks, yt, zita, nodeID):
        N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
        N = self.Shapefunc.fun(ks, yt, zita)
        NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
        xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
        J = np.dot(NksNytNzita, xyz.T)
        fx, fy, fz = 0, 0, 0
        if abs(ks) == 1:
            fx = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) * N[nodeID]
            fy = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) * N[nodeID]
            fz = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) * N[nodeID]
        elif abs(yt) == 1:
            fx = (J[2, 1] * J[0, 2] - J[2, 2] * J[0, 1]) * N[nodeID]
            fy = (J[2, 2] * J[0, 0] - J[2, 0] * J[0, 2]) * N[nodeID]
            fz = (J[2, 0] * J[0, 1] - J[2, 1] * J[0, 0]) * N[nodeID]
        elif abs(zita) == 1:
            fx = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) * N[nodeID]
            fy = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) * N[nodeID]
            fz = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) * N[nodeID]
        return fx, fy, fz
