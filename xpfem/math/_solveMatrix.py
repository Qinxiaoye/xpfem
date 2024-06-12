import scipy as sci

class solveMatrix:
    def __init__(self,A,b) -> None:
        self.A = A
        self.b = b

    def solveByScipy(self):
        x = sci.sparse.linalg.spsolve(self.A, self.b)

        return x

