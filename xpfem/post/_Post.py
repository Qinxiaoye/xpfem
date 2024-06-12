import pyvista as pv
import numpy as np
import vtkmodules.all as vtk

def calculateSum(u):
    ndim = u.shape[1]
    if ndim == 2:
        usum = (u[:,0]**2+u[:,1]**2)**0.5
    elif ndim == 3:
        usum = (u[:,0]**2+u[:,1]**2+u[:,2]**2)**0.5
    return usum

def symMesh(mesh,sym):
    if not isinstance(sym,list):
        sym = [sym]
    for n in range(0,len(sym)):
        if sym[n] == 'x':
            mesh += mesh.reflect((1, 0, 0), point=(0, 0, 0))
        elif sym[n] == 'y':
            mesh += mesh.reflect((0, 1, 0), point=(0, 0, 0))
        elif sym[n] == 'z':
            mesh += mesh.reflect((0, 0, 1), point=(0, 0, 0))

    return mesh


class Post:
    def __init__(self,ndim,mnode):
        self.ndim = ndim
        self.mnode = mnode

    def showScalar(self,region,scalar,name = 'scalar',showEdge=True):
        # 显示标量
        sumElem = region.mesh.sumElem
        node = region.mesh.points
        elem = region.mesh.cells
        celltypes = np.empty(sumElem, dtype=np.uint8)
        if self.ndim == 2:
            if self.mnode == 4:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
        else:
            celltypes[:] = vtk.VTK_HEXAHEDRON  # 8节点三维单元

        if self.ndim == 2:
            elemNew = elem[:,0:4]
        else:
            elemNew = elem[:,0:8]
        head = np.ones((elemNew.shape[0], 1), int)*elemNew.shape[1]

        dargs = dict(
            cmap="coolwarm",
            show_scalar_bar=True,
            scalar_bar_args={'title': name,'color':'k'},
        )
        if self.ndim == 2:
            node=np.concatenate((node,np.zeros((node.shape[0],1),'float')),axis=1)
        mesh = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node)
        
        pl = pv.Plotter()
        pl.add_mesh(mesh, scalars=scalar, **dargs,show_edges=showEdge)
        pl.background_color = 'white'
        pl.add_camera_orientation_widget()
        pl.add_text("sumNode = "+str(region.mesh.sumNode)+"\n sumElem = "+str(region.mesh.sumElem), color='k')
        pl.show()

    def showDisplacement(self,region,displacement,direction = 'all',sym = 'none',showEdge=True,
                        show_deformed = False, scale = 1.0, show_shade = False,opacity = 0.2):
        # 显示位移，二维和三维
        sumElem = region.mesh.sumElem
        node = region.mesh.points
        elem = region.mesh.cells
        celltypes = np.empty(sumElem, dtype=np.uint8)
        if self.ndim == 2:
            if self.mnode == 4:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
            elif self.mnode == 8:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
        else:
            celltypes[:] = vtk.VTK_HEXAHEDRON  # 8节点三维单元

        if show_deformed:
            node_deformed = node+np.reshape(displacement,(-1,self.ndim))*scale
            if self.ndim == 2:
                node_deformed=np.concatenate((node_deformed,np.zeros((node.shape[0],1),'float')),axis=1)

        if self.ndim == 2:
            elemNew = elem[:,0:4]
            ux = displacement[:,0]
            uy = displacement[:,1]
        else:
            elemNew = elem[:,0:8]
            ux = displacement[:,0]
            uy = displacement[:,1]
            uz = displacement[:,2]
        head = np.ones((elemNew.shape[0], 1), int)*elemNew.shape[1]
        if self.ndim == 2:
            node=np.concatenate((node,np.zeros((node.shape[0],1),'float')),axis=1)

        if show_deformed:
            mesh = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node_deformed)
        else:
            mesh = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node)

        mesh.point_data['x'] = ux
        mesh.point_data['y'] = uy
        if self.ndim == 3:
            mesh.point_data['z'] = uz
        usum = calculateSum(np.reshape(displacement,(-1,self.ndim)))
        mesh.point_data['sum'] = usum

        mesh = symMesh(mesh,sym)

        dargs = dict(
            cmap="coolwarm",
            show_scalar_bar=True,
            scalar_bar_args={'title': direction,'color':'k'},
        )
        
        if direction == 'all':
            dargs = dict(
            cmap="coolwarm",
            show_scalar_bar=False,
            )
            if self.ndim == 2:
                pl = pv.Plotter(shape=(1, 2))
                pl.subplot(0, 0)
                mesh.point_data.active_scalars_name = 'x'
                pl.add_mesh(mesh.copy(),  **dargs,show_edges=showEdge)
                pl.add_text("X Displacement", color='k')
                pl.subplot(0, 1)
                mesh.point_data.active_scalars_name = 'y'
                pl.add_mesh(mesh.copy(), **dargs,show_edges=showEdge)
                pl.add_text("Y Displacement", color='k')
                pl.view_xy()
                pl.link_views()
            else:
                pl = pv.Plotter(shape=(1, 3))
                pl.subplot(0, 0)
                mesh.point_data.active_scalars_name = 'x'
                pl.add_mesh(mesh.copy(), **dargs,show_edges=showEdge)
                pl.add_text("X Displacement", color='k')
                pl.subplot(0, 1)
                mesh.point_data.active_scalars_name = 'y'
                pl.add_mesh(mesh.copy(), **dargs,show_edges=showEdge)
                pl.add_text("Y Displacement", color='k')
                pl.subplot(0, 2)
                mesh.point_data.active_scalars_name = 'z'
                pl.add_mesh(mesh.copy(), **dargs,show_edges=showEdge)
                pl.add_text("Z Displacement", color='k')
                pl.view_xy()
                pl.link_views()
        elif direction == 'x':
            mesh.point_data.active_scalars_name = 'x'
            pl = pv.Plotter()
            pl.add_mesh(mesh, **dargs,show_edges=showEdge)
            pl.background_color = 'white'
            pl.add_text("X Displacement", color='k')

        elif direction == 'y':
            mesh.point_data.active_scalars_name = 'y'
            pl = pv.Plotter()
            pl.add_mesh(mesh, **dargs,show_edges=showEdge)
            pl.background_color = 'white'
            pl.add_text("Y Displacement", color='k')

        elif direction == 'z':
            mesh.point_data.active_scalars_name = 'z'
            pl = pv.Plotter()
            pl.add_mesh(mesh, **dargs,show_edges=showEdge)
            pl.background_color = 'white'
            pl.add_text("Z Displacement", color='k')

        elif direction == 'sum':
            mesh.point_data.active_scalars_name = 'sum'
            pl = pv.Plotter()
            pl.add_mesh(mesh, **dargs,show_edges=True)
            pl.background_color = 'white'
            pl.add_text("Total Displacement", color='k')

        if show_shade:
            mesh_undeformed = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node)
            mesh_undeformed = symMesh(mesh_undeformed,sym)
            pl.add_mesh(mesh_undeformed,color='w',opacity=opacity)

        pl.background_color = 'white'
        pl.add_camera_orientation_widget()
        if self.ndim == 2:
            pl.view_xy()
        pl.show()

    def getDisplacementMatrix(self,displacement):
        solution = np.reshape(displacement,(-1,self.ndim))

        return solution
    

    def showStress(self,region,displacement,stress,mises,direction = 'sxx',sym = 'none',showEdge=True,
                        show_deformed = False, scale = 1.0, show_shade = False,opacity = 0.2):
        # show stresses, for 2D and 3D
        sumElem = region.mesh.sumElem
        node = region.mesh.points
        elem = region.mesh.cells
        celltypes = np.empty(sumElem, dtype=np.uint8)
        if self.ndim == 2:
            if self.mnode == 4:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
            elif self.mnode == 8:
                celltypes[:] = vtk.VTK_QUAD # 4节点二维单元
        else:
            celltypes[:] = vtk.VTK_HEXAHEDRON  # 8节点三维单元

        if show_deformed:
            node_deformed = node+np.reshape(displacement,(-1,self.ndim))*scale
            if self.ndim == 2:
                node_deformed=np.concatenate((node_deformed,np.zeros((node.shape[0],1),'float')),axis=1)

        if self.ndim == 2:
            elemNew = elem[:,0:4]
            ux = displacement[0::2]
            uy = displacement[1::2]
        else:
            elemNew = elem[:,0:8]
            ux = displacement[0::3]
            uy = displacement[1::3]
            uz = displacement[2::3]
        head = np.ones((elemNew.shape[0], 1), int)*elemNew.shape[1]
        if self.ndim == 2:
            node=np.concatenate((node,np.zeros((node.shape[0],1),'float')),axis=1)

        if show_deformed:
            mesh = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node_deformed)
        else:
            mesh = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node)

        if self.ndim == 2:
            map = {'sxx':0,'syy':1,'sxy':2,"syx":2}
        else:
            map = {'sxx':0,'syy':1,'szz':2,"sxy":3,'sxz':4,'syz':5}

        direction = direction.lower()
        if direction == 'mises':
            mesh.point_data[direction] = mises
        else:
            mesh.point_data[direction] = stress[:,map[direction]]
        

        mesh = symMesh(mesh,sym)

        dargs = dict(
            cmap="coolwarm",
            show_scalar_bar=True,
            scalar_bar_args={'title': direction,'color':'k'},
        )
        
        pl = pv.Plotter()
        pl.add_mesh(mesh, **dargs,show_edges=showEdge)
        pl.background_color = 'white'
        mesh.point_data.active_scalars_name = direction
        pl.add_text(direction, color='k')
        

        if show_shade:
            mesh_undeformed = pv.UnstructuredGrid(np.hstack((head, elemNew)), celltypes, node)
            mesh_undeformed = symMesh(mesh_undeformed,sym)
            pl.add_mesh(mesh_undeformed,color='w',opacity=opacity)

        pl.background_color = 'white'
        pl.add_camera_orientation_widget()
        if self.ndim == 2:
            pl.view_xy()
        pl.show()