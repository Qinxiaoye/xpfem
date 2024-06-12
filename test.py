import numpy as np
import xpfem as fem

# read the mesh
dir = 'input\\3D'
mesh=fem.mesh.readtxt(dir+'\\'+'NLIST.dat',dir+'\\'+'ELIST.dat',dim=3)
# Define Sets
set = {'set1':np.arange(0,mesh.sumElem)}

# Define Material
mat1 = fem.LinearElastic(E=200000,nu=0.3)
part = {'set1':mat1} # part 与 set对应
umat = fem.CreatMat(part,set)

ndim, mnode = mesh.ndim,mesh.mnode
trcounter = 0

# 有限元区域
region = fem.Region(mesh,umat,trcounter)
# 压力边界条件
Numman = fem.transPress(mnode, ndim)
Numman.loadPress(filename=dir+'\\'+'press.dat')
Numman.transpress(mesh.points,mesh.cells)

# 位移边界条件
fix = fem.fixBoundary(ndim,method = 'change_1')
fix.loadFix(filename=dir+'\\'+'fixNode.dat')
fix.fixValue()

# 创建FEM求解
Fe = fem.FEM_Solver_linearElasticity()
Fe.FEM_steady(region,Numman,fix) # 也可以一步步求解，不用FEM_steady方法

# 后处理
post = fem.Post(ndim,mnode)
# post.showDisplacement(region,Fe.u,direction = 'y',show_deformed = False, scale = 1.0, show_shade = True, opacity = 0.2) # direction: x,y,z,sum,all

post.showStress(region,Fe.u,Fe.stress,Fe.mises,direction = 'mises') # direction: x,y,z,sum,all

file = fem.writeResult()
varName =['ux','uy','uz','sxx','syy','szz','sxy','syz','sxz','von-Mises']
file.writeSolution(mesh,np.c_[Fe.u,Fe.stress,Fe.mises],varName,fileName='solution.bina')
file.writeVTK(mesh,np.c_[Fe.u,Fe.stress,Fe.mises],varName,fileName='solution.vtk')
