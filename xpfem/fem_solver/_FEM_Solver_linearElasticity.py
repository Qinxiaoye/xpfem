from ._FEM_Solve import FEM_Solve
from xpfem import linearElasticity,math,solvestress
from xpfem.boundary import boundaryCondition
import numpy as np
# from line_profiler import LineProfiler

class FEM_Solver_linearElasticity(FEM_Solve):
    def __init__(self,calculateStress = True):
        self.calculateStress = calculateStress
        pass
        

    def FEM_steady(self,region,Numman,fix):
        mnode = region.mnode
        ndim = region.ndim
        sumNode = region.mesh.sumNode
        if ndim == 2:
            globalK = linearElasticity.GlobalK2D(mnode)
        else:
            globalK = linearElasticity.GlobalK3D(mnode)

        GK_u, GK_v, GK_a = globalK.GK(region) # region.mesh.cells, region.mesh.points, region.umat

        nodeForce = Numman.nodeForce

        # profile = LineProfiler(globalK.GK)
        # profile.runcall(globalK.GK, region)
        # profile.print_stats()
        
        # 第一类边界条件
        bc = boundaryCondition()
        if fix.method == 'change_1':
            GK, force = bc.boundary_1(GK_u, GK_v, GK_a, nodeForce, fix.dof, sumNode*ndim)

        # 采用scipy求解
        solve = math.solveMatrix(GK,force)
        displacement = solve.solveByScipy()

        # 求解应力
        if self.calculateStress:
            if ndim == 2:
                Stress = solvestress.getStress2D(mnode)
            else:
                Stress = solvestress.getStress3D(mnode)
            stress, mises = Stress.evualStress(displacement, region)
        self.stress = stress
        self.mises = mises

        displacement = np.reshape(displacement,(-1,ndim))
        super().__init__(GK, force,displacement)


        # super().__init__(mesh, element, quadrature)
    

    # def FEM_steady(self, node, elem, Ex, mu, GlobalK, nodeForce,fixNode):
    #     sumNode = node.shape[0]
    #     GK_u, GK_v, GK_a = GlobalK.GK(elem, node, Ex, mu)
    #     GK, force = self.boundary_cond(GK_u, GK_v, GK_a, nodeForce, fixNode, sumNode)
    #     displacement = sci.sparse.linalg.spsolve(GK, force)
    #     return displacement

    # def FEM_dynamic(self, node, elem, Ex, mu, GlobalK, GM, nodeForce,fixNode, Tr, alpha1, alpha2):
    #     sumNode = node.shape[0]
    #     GK_u, GK_v, GK_a = GlobalK.GK(elem, node, Ex, mu)
    #     GK, force = self.boundary_cond(GK_u, GK_v, GK_a, nodeForce, fixNode, sumNode)
    #     C = alpha1 * GM + alpha2 * GK
    #     displacement = Tr.Newmark(sumNode, GK, GM, C, force)
    #     return displacement