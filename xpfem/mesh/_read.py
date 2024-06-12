import numpy as np


from ._mesh import Mesh


def readtxt(
        nodeFileName='NLIST.dat', elemFileName='ELIST.dat',dim=None,delimiter=None
):
    from numpy import loadtxt
    node=loadtxt(nodeFileName,delimiter=delimiter)
    elem=loadtxt(elemFileName,delimiter=delimiter)

    node = node[:,1:]

    if dim is None:
        dim = node.shape[1]

    if elem.ndim ==1:
        elem = elem.reshape(1,-1)
    node = node[:,:dim]
    elem = elem[:,2:]

    elem = elem - 1
    mnode = elem.shape[1]
    mesh = Mesh(node,elem,mnode,dim)
    return mesh