import numpy as np
import xpfem as fem

# read the mesh
dir = 'input\\pipe'
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
Fe_analysis = fem.FEM_Solver_linearElasticity()
Fe_analysis.FEM_steady(region,Numman,fix) # 也可以一步步求解，不用FEM_steady方法

# 后处理
post = fem.Post(ndim,mnode)
# post.showDisplacement(region,Fe_analysis.u,direction = 'sum') # direction: x,y,z,sum,all
post.showDisplacement(region,Fe_analysis.u,direction = 'sum',show_deformed = True, scale = 5.0, show_shade = False, opacity = 0.1) # direction: x,y,z,sum,all
solution = post.getDisplacementMatrix(Fe_analysis.u)
file = fem.writeResult('solution.bina')
file.writeSolution(mesh.points,mesh.cells,ndim,solution,['ux','uy','uz'])

# ux = Fe_analysis.u[0::3]
# uy = Fe_analysis.u[1::3]
# uz = Fe_analysis.u[2::3]
# post.showScalar(region,uy,name='uy')
