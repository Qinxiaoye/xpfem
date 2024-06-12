import numpy as np
from xpfem import shapefunction


class transPress:

    def __init__(self, mnode, ndim):
        self.Dfun = shapefunction.Dfun(mnode)
        self.Shapefunc = shapefunction.Shapefunc(mnode, ndim)
        self.mnode = mnode
        self.ndim = ndim

    def loadPress(self,filename = 'press.dat'):
        press = np.loadtxt(filename)
        if press.ndim == 1:
            press = press.reshape(1,-1)
        self.press = press

    def transpress(self, node, elem):
        sumPre = self.press.shape[0]
        sumNode = node.shape[0]
        nodeForce = np.zeros([sumNode,self.ndim])

        elemP = self.press[:, 0]
        face = self.press[:, 1]
        pres = self.press[:, 2]

        elemP = elemP - 1
        face = face - 1
        elemP = elemP.astype('int')
        face = face.astype('int')

        nip = 3
        ks = np.array([-0.774596669241483, 0, 0.774596669241483])
        w = np.array([0.55555555555556, 0.888888888888888889, 0.55555555555556])

        if self.ndim == 2:
            if self.mnode == 4:
                faceNode = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
            else:
                faceNode = np.array([[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]], dtype=int)

        else:
            if self.mnode == 8:
                faceNode = np.array([[1,4,3,2], [1,2,6,5], [2,3,7,6], [3,4,8,7], [1,5,8,4], [5,6,7,8]]).astype('int')-1
            else:
                faceNode = np.array([[1,4,3,2,9,12,11,10], [1,2,6,5,9,18,13,17],
                                    [2, 3, 7, 6, 10, 19, 14, 18], [3, 4, 8, 7, 11, 20, 15, 19],
                                    [1, 5, 8, 4, 17, 16, 20, 12], [5, 6, 7, 8, 13, 14, 15, 16]]).astype('int')-1
                
        if self.ndim == 2:
            for i in range(1, sumPre + 1):
                bNode = elem[elemP[i - 1], faceNode[face[i - 1], :]]
                if self.mnode == 4:
                    p = 0.5 * pres[i - 1] *np.array([node[bNode[0], 1] - node[bNode[1], 1],node[bNode[1], 0] - node[bNode[0], 0]])
                elif self.mnode == 8:
                    p = pres[i - 1]*np.array([
                        1 / 2 * node[bNode[0], 1] + 1 / 6 * node[bNode[1], 1] - 2 / 3 * node[bNode[2], 1],
                        -(1 / 2 * node[bNode[0], 0] + 1 / 6 * node[bNode[1], 0] - 2 / 3 * node[bNode[2], 0]),
                        -(1 / 6 * node[bNode[0], 1] + 1 / 2 * node[bNode[1], 1] - 2 / 3 * node[bNode[2], 1]),
                        1 / 6 * node[bNode[0], 0] + 1 / 2 * node[bNode[1], 0] - 2 / 3 * node[bNode[2], 0],
                        2/3*node[bNode[0], 1] - 2/3*node[bNode[1], 1],
                        -(2/3*node[bNode[0], 0] - 2/3*node[bNode[1], 0])
                    ])
                
                if self.mnode == 4:
                    nodeForce[bNode,:] = nodeForce[bNode,:]-p
                elif self.mnode == 8:
                    nodeForce[bNode,:] = nodeForce[bNode,:]-np.reshape(p,[3,-1])

        if self.ndim == 3:      
            normal = np.array([
                [0,0,-1],
                [1,0,0],
                [0,1,0],
                [-1,0,0],
                [0,-1,0],
                [0,0,1]
            ])
            for n in range(1,sumPre+1):
                press = np.zeros((self.mnode,self.ndim))
                x = node[elem[elemP[n - 1], :], 0]
                y = node[elem[elemP[n - 1], :], 1]
                z = node[elem[elemP[n - 1], :], 2]

                for i in range(1, nip+1):
                    for j in range(1, nip+1):
                        if face[n - 1] == 0:
                            xi,eta,zeta = ks[i - 1], ks[j - 1], -1
                        elif face[n - 1] == 1:
                            xi,eta,zeta = 1, ks[i - 1], ks[j - 1]
                        elif face[n - 1] == 2:
                            xi,eta,zeta = ks[i - 1], 1, ks[j - 1]
                        elif face[n - 1] == 3:
                            xi,eta,zeta = -1, ks[i - 1], ks[j - 1]
                        elif face[n - 1] == 4:
                            xi,eta,zeta = ks[i - 1], -1, ks[j - 1]
                        elif face[n - 1] == 5:
                            xi,eta,zeta = ks[i - 1], ks[j - 1], 1
                        
                        adjoint_J = self.calculateJ( x, y, z, xi,eta,zeta)
                        shapeFunc = self.Shapefunc.fun(xi,eta,zeta)
                        press = press+w[i-1]*w[j-1]*np.outer(shapeFunc,np.dot(normal[face[n - 1],:],adjoint_J.T))

                # press = press.flatten()
                elemNode = elem[elemP[n - 1], :]
                nodeForce[elemNode,:] = nodeForce[elemNode,:]+press*pres[n - 1]

        nodeForce = -nodeForce.reshape(-1)
        self.nodeForce = nodeForce

    def calculateJ(self, x, y, z, ks, yt, zita):
        N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
        NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
        xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
        J = np.dot(NksNytNzita, xyz.T)
        adjoint_J = np.array(
        [
            [J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1], J[2, 1] * J[0, 2] - J[2, 2] * J[0, 1], J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]],
            [J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2], J[2, 2] * J[0, 0] - J[2, 0] * J[0, 2], J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]],
            [J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0], J[2, 0] * J[0, 1] - J[2, 1] * J[0, 0], J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]]
        ]
        )
        return adjoint_J