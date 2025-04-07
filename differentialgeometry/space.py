import sympy as sp
import numpy as np

class Space:
    def __init__(self, G:sp.Matrix, X:sp.Matrix):
        self.metricG = G
        self.X = X
        self.dim = len(X)
        self.chris = self.getChristoffelSymbols()
        self.coefficientsR = self.getCoefficientFunctionsR()
    
    def getChristoffelSymbols(self):
        """
        Uses the metric tensor to compute the christoffel symbols.
        The symbols are stored in a 3D numpy array with indices (i,j,m).
        """
        chris = sp.zeros(self.dim, self.dim, self.dim)

        for i in range(self.dim):
            chris[i] = sp.Matrix(chris[i])

        for i in range(self.dim):
            for j in range(self.dim):
                for m in range(self.dim):
                    chris_i = 0
                    for l in range(self.dim):
                        chris_i += sp.trigsimp((0.5 * (
                                (self.metricG[j, l].diff(self.X[i]))
                                + self.metricG[l, i].diff(self.metricG[j])
                                + self.metricG[i, j].diff(self.X[l]))))
                    chris[i,j,m] = chris_i * sp.trigsimp(self.metricG.inv())[m,i]
        return chris
    
    def metricTensorg(self, V, W):
        """
        gp(V, W) = V^T * G * W
        where V and W are vectors in the tangent space TpU.
        This function returns a sympy matrix containing the metric tensor.
        """
        return V.T * self.metricG * W

    def getCoefficientFunctionsR(self):
        """
        Uses the christoffel symbols to compute the coefficients of the Riemann curvature tensor.
        The coefficients are stored in a 4D numpy array with indices (i,j,k,m).
        """
        R = np.zeros((self.dim, self.dim, self.dim, self.dim), dtype=object)

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for m in range(self.dim):
                        R[i, j, k, m] = (self.chris[j,k,m].diff(self.X[i]) - 
                                        self.chris[i,k,m].diff(self.X[j])
                                        )
                        for s in range(self.dim):
                            R[i, j, k, m] += (
                                self.chris[j,k,s]*self.chris[i,s,m] - 
                                self.chris[i,k,s]*self.chris[j,s,m]
                                )
        return R