import pyvista as pv
import numpy as np
import vtkmodules.all as vtk
import struct
from ._fileBinaClass import fileBinaClass

class writeResult:
    def __init__(self):
        pass

    def writeSolution(self,mesh,solution,varName = [],fileName='solution.bina'):
        writeFile = fileBinaClass()
        node = mesh.points
        elem = mesh.cells
        ndim = mesh.ndim
        sumNode = node.shape[0]
        sumElem = elem.shape[0]
        sumVar = solution.shape[1]
        fid = open(fileName,'wb')
        fid.write(struct.pack('4i',ndim,sumVar+ndim,sumNode,sumElem))

        writeFile.write_string(fid,'x')
        writeFile.write_string(fid,'y')
        if ndim == 3:
            writeFile.write_string(fid,'z')

        if varName.__len__() == 0 or varName.__len__()!= sumVar:
            for n in range(0,sumVar):
                name = 'v'+str(n+1)
                writeFile.write_string(fid,name)
        else:
            for name in varName:
                writeFile.write_string(fid,name)
        # 输出节点
        writeFile.write_matrix(fid,node,'float')
        # 输出单元
        writeFile.write_matrix(fid,elem+1,'int')

        # 输出结果
        writeFile.write_matrix(fid,solution,'float')
        


        fid.close()
    
    def writeVTK(self,mesh,solution,varName = [],fileName='solution.vtk'):
        node = mesh.points
        elem = mesh.cells
        celltypes = np.empty(mesh.sumElem, dtype=np.uint8)
        if mesh.ndim == 2:
            if mesh.mnode == 4:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
            elif mesh.mnode == 8:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
        else:
            celltypes[:] = vtk.VTK_HEXAHEDRON  # 8节点三维单元
        
        if mesh.ndim == 2:
            elemNew = elem[:,0:4]
            node=np.concatenate((node,np.zeros((node.shape[0],1),'float')),axis=1)
        else:
            elemNew = elem[:,0:8]
        head = np.ones((elemNew.shape[0], 1), int)*elemNew.shape[1]
        mesh = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node)

        sumVar = solution.shape[1]
        n = 0
        if varName.__len__() == 0 or varName.__len__()!= sumVar:
            for n in range(0,sumVar):
                name = 'v'+str(n+1)
                mesh.point_data[name] = solution[:,n]
        else:
            for name in varName:
                mesh.point_data[name] = solution[:,n]
                n = n+1
        
        mesh.save(fileName)