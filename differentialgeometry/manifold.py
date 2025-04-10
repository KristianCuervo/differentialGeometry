import sympy as sp
import numpy as np


class RiemannianManifold:
    def __init__(self, vars:sp.Matrix, metric:sp.Matrix=None):
        """
        Generic n-dimensional manifold with optional metric.
        Used as a superclass.
        vars = coordinates in manifold
        metric = metric matrix G of manifold (if given).
        """

        self.vars = vars
        self.dim = len(vars)
        self.metric = metric
        self.metric_inv = None
        if metric.det() != 0:
            self.metric_inv = metric.inv()
        
        # Precompute geometry
        self.christoffels = self._compute_christoffel_symbols


    def metric_tensor(self, V:sp.Matrix, W:sp.Matrix):
        """
        g(V, W) = V^T * metric * W
        """
        return (V.T * self.metric * W)[0, 0]
    
    def _compute_christoffel_symbols(self):
        """
        Returns a sympy 3D array of Γ^m_{ij}.
        Access using christoffel[i,j,m]
        """
        chris = sp.zeros(self.dim, self.dim, self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                for m in range(self.dim):
                    term = 0
                    for l in range(self.dim):
                        term += 0.5 * (
                            self.metric[j, l].diff(self.coord_vars[i]) +
                            self.metric[l, i].diff(self.coord_vars[j]) -
                            self.metric[i, j].diff(self.coord_vars[l])
                        ) * self.metric_inv[m, l]
                    chris[i, j, m] = sp.simplify(term)
        return chris

    def _compute_riemann_tensor(self) -> np.array:
        """
        Returns a 4D numpy array R^m_{ijk}.
        Access using R[i,j,k,m]
        """
        R = np.zeros((self.dim, self.dim, self.dim, self.dim), dtype=object)
        chris = self.christoffels
        X = self.coord_vars
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for m in range(self.dim):
                        term = chris[j, k, m].diff(X[i]) - chris[i, k, m].diff(X[j])
                        for s in range(self.dim):
                            term += chris[j, k, s]*chris[i, s, m] - chris[i, k, s]*chris[j, s, m]
                        R[i, j, k, m] = sp.simplify(term)
        return R