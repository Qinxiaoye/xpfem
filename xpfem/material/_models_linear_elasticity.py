import numpy as np

def lame_converter(E, nu):
    """Convert material parameters to first and second Lamé - constants.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    Returns
    -------
    lmbda : float
        First Lamé - constant.
    mu : float
        Second Lamé - constant (shear modulus).
    """

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    return lmbda, mu

class LinearElastic:
    def __init__(self, E=None, nu=None):
        self.E = E
        self.nu = nu
        self.elasticity = self.hessian

    def hessian(self, E=None,nu = None):
        """Evaluate the 3D elasticity tensor from the deformation gradient.

        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (6x6)

        """

        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu
        d0 = (E*(1-nu))/((1+nu)*(1-2*nu))
        d1 = nu/(1-nu)
        d2=(1-2*nu)/(2*(1-nu))

        elast = np.array([[1,d1,d1,0,0,0],
                          [d1,1,d1,0,0,0],
                          [d1,d1,1,0,0,0],
                          [0,0,0,d2,0,0],
                          [0,0,0,0,d2,0],
                          [0,0,0,0,0,d2]])
        return d0*elast
    


class LinearElasticPlaneStrain:
    """Plane-strain isotropic linear-elastic material formulation.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.elasticity = self.hessian

    def hessian(self, E=None,nu = None):
        """Evaluate the 2D plane strain elasticity tensor from the deformation gradient.

        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (3x3)

        """    
        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # for plane strain 
        e0 = E/(1-nu**2)
        nu0 = nu/(1-nu)

        d0 = e0/(1-nu0**2)
        d1 = nu0
        d2=(1-nu0)/2

        elast = np.array([[1,d1,0],
                          [d1,1,0],
                          [0,0,d2]])
        return d0*elast

class LinearElasticPlaneStress:
    """Plane-stress isotropic linear-elastic material formulation.

    Arguments
    ---------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    """

    def __init__(self, E, nu, name=None):
        self.E = E
        self.nu = nu
        self.name=name
        self.elasticity = self.hessian

    def hessian(self, E=None,nu = None):
        """Evaluate the 2D plane strain elasticity tensor from the deformation gradient.

        E : float, optional
            Young's modulus (default is None)
        nu : float, optional
            Poisson ratio (default is None)

        Returns
        -------
        ndarray
            In-plane components of elasticity tensor (3x3)

        """    
        if E is None:
            E = self.E

        if nu is None:
            nu = self.nu

        # for plane stress 
        e0 = E
        nu0 = nu

        d0 = e0/(1-nu0**2)
        d1 = nu0
        d2=(1-nu0)/2

        elast = np.array([[1,d1,0],
                          [d1,1,0],
                          [0,0,d2]])
        return d0*elast