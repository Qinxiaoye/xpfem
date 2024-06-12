import numpy as np
from xpfem import shapefunction,math
# from line_profiler import LineProfiler


class GlobalK3D:
    def __init__(self, mnode):
        self.mnode = mnode
        self.ndim = 3
        self.Dfun= shapefunction.Dfun(mnode)

    def GK(self, region):
        elem = region.mesh.cells
        node = region.mesh.points
        umat = region.umat

        sumElem = elem.shape[0]

        nnz0 = (self.mnode*3)*(self.mnode*3)
        nnz = sumElem*nnz0

        
        u = np.zeros(nnz,dtype='int64')
        v = np.zeros(nnz,dtype='int64')
        a = np.zeros(nnz)
        x = np.zeros((sumElem, self.mnode))
        y = np.zeros((sumElem, self.mnode))
        z = np.zeros((sumElem, self.mnode))
        for m in range(1, self.mnode + 1):
            x[:, m - 1] = node[elem[:, m - 1], 0]
            y[:, m - 1] = node[elem[:, m - 1], 1]
            z[:, m - 1] = node[elem[:, m - 1], 2]

        # 高斯积分
        self.gauss_int = math.gauss_legendre(intOrder=2,dim=3)

        # 按照单元进行循环
        ID = 0
        for name in umat.nameList:
            D = umat.part[name].hessian()

            for elemID in umat.set[name]:
                ID = ID+1
                if self.mnode == 8:
                    K = self.elemK3D_bbar(x[elemID, :], y[elemID, :], z[elemID, :], D)
                else:
                    K = self.elemK3D(x[elemID, :], y[elemID, :], z[elemID, :], D)
                nodeID = elem[elemID, :]
                ndfID = np.zeros(3 * self.mnode)
                ndfID[0::3] = nodeID * 3
                ndfID[1::3] = nodeID * 3 + 1
                ndfID[2::3] = nodeID * 3 + 2
                ddd = np.tile(ndfID, (3 * self.mnode, 1))
                u[nnz0*(ID-1):nnz0*ID] = ddd.flatten('F')
                v[nnz0*(ID-1):nnz0*ID] = np.tile(ndfID, 3 * self.mnode)
                a[nnz0*(ID-1):nnz0*ID] = K.flatten('F')
                
        # profile = LineProfiler(self.elemK3D)
        # profile.runcall(self.elemK3D, x[elemID, :], y[elemID, :], z[elemID, :], D)
        # profile.print_stats()

        return u, v, a

    def elemK3D(self, x, y, z, D):
        Ke = np.zeros((self.mnode * self.ndim, self.mnode * self.ndim))
        for i in range(len(self.gauss_int.points)):
            xi, eta, zeta = self.gauss_int.points[i]
            weight = self.gauss_int.weights[i]
            J, B = self.elemB3D(x, y, z, xi,eta,zeta)
            Ke = Ke + weight * np.dot(np.dot(B.T, D), B) * np.linalg.det(J)

        return Ke
        
    def elemB3D(self, x, y, z, ks, yt, zita):
        B = np.zeros((6, 3 * self.mnode))
        N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
        NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
        xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
        J = np.dot(NksNytNzita, xyz.T)
        N_cor = np.linalg.solve(J, NksNytNzita)
        for n in range(1, self.mnode+1):
            B[0, 3 * n - 3] = N_cor[0,n-1]
            B[1, 3 * n - 2] = N_cor[1,n-1]
            B[2, 3 * n - 1] = N_cor[2,n-1]
            B[3, 3 * n - 3] = N_cor[1,n-1]
            B[3, 3 * n - 2] = N_cor[0,n-1]
            B[4, 3 * n - 2] = N_cor[2,n-1]
            B[4, 3 * n - 1] = N_cor[1,n-1]
            B[5, 3 * n - 3] = N_cor[2,n-1]
            B[5, 3 * n - 1] = N_cor[0,n-1]
        return J, B
    
    def elemK3D_bbar(self, x, y, z, D):
        # bbar技术计算单元刚度矩阵，适用于三维8节点线弹性力学
        Ke = np.zeros((self.mnode * self.ndim, self.mnode * self.ndim))
        for i in range(len(self.gauss_int.points)):
            xi, eta, zeta = self.gauss_int.points[i]
            weight = self.gauss_int.weights[i]
            J,Bdev = self.Bdev(x, y, z, xi,eta,zeta)
            Ke = Ke + weight * np.dot(np.dot(Bdev.T, D), Bdev) * np.linalg.det(J)
        # 体积应变的减缩积分
        # 积分点坐标 xi, eta, zeta = 0,0,0,权重为 weight = 8
        xi, eta, zeta = 0.0, 0.0, 0.0
        weight = 8.0
        J,Bvol = self.elemB3Dvol( x, y, z, xi, eta, zeta)
        Ke = Ke + weight * np.dot(np.dot(Bvol.T, D), Bvol) * np.linalg.det(J)

        return Ke
    
    
    def elemB3Dvol(self, x, y, z, ks, yt, zita):
        Bvol = np.zeros((6, 3 * self.mnode))
        N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
        NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
        xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
        J = np.dot(NksNytNzita, xyz.T)
        N_cor = np.linalg.solve(J, NksNytNzita)

        for n in range(1, self.mnode+1):
            Bvol[0, 3 * n - 3] = N_cor[0,n-1]
            Bvol[0, 3 * n - 2] = N_cor[1,n-1]
            Bvol[0, 3 * n - 1] = N_cor[2,n-1]
            Bvol[1, 3 * n - 3] = N_cor[0,n-1]
            Bvol[1, 3 * n - 2] = N_cor[1,n-1]
            Bvol[1, 3 * n - 1] = N_cor[2,n-1]
            Bvol[2, 3 * n - 3] = N_cor[0,n-1]
            Bvol[2, 3 * n - 2] = N_cor[1,n-1]
            Bvol[2, 3 * n - 1] = N_cor[2,n-1]
        Bvol = Bvol/3.0
        return J, Bvol
    

    def Bdev(self, x, y, z, ks, yt, zita):
        J,B = self.elemB3D2( x, y, z, ks, yt, zita)
        J,Bvol = self.elemB3Dvol( x, y, z, ks, yt, zita)
        return J, (B - Bvol)
    
    def elemB3D2(self, x, y, z, ks, yt, zita):
        # 用于计算bbar单元
        B = np.zeros((6, 3 * self.mnode))
        N_ks, N_yt, N_zita = self.Dfun.dfun3D(ks, yt, zita)
        NksNytNzita = np.concatenate((N_ks.T, N_yt.T, N_zita.T))
        xyz = np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
        J = np.dot(NksNytNzita, xyz.T)
        N_cor = np.linalg.solve(J, NksNytNzita)
        for n in range(1, self.mnode+1):
            B[0, 3 * n - 3] = N_cor[0,n-1]
            B[1, 3 * n - 2] = N_cor[1,n-1]
            B[2, 3 * n - 1] = N_cor[2,n-1]
            B[3, 3 * n - 3] = N_cor[1,n-1]
            B[3, 3 * n - 2] = N_cor[0,n-1]
            B[4, 3 * n - 3] = N_cor[2,n-1]
            B[4, 3 * n - 1] = N_cor[0,n-1]
            B[5, 3 * n - 2] = N_cor[2,n-1]
            B[5, 3 * n - 1] = N_cor[1,n-1]
        return J, B