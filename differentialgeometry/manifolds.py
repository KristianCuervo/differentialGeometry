import sympy as sp

class Surface:
    def __init__(self, expr:sp.Matrix, var:sp.Matrix):
        self.expr = expr
        self.var = var
        self.dim = len(var)
        self.metricG = self.metricMatrixG()
    
    def metricMatrixG(self):
        """
        Computes the metric matrix G from the parameterization r.
        G = J^T * J
        where J is the Jacobian of r with respect to X.
        """
        rJ = self.expr.jacobian(self.var)
        return rJ.T * rJ
    

class Curve:
    def __init__(self, gamma:sp.Matrix, t:sp.symbols):
        self.expr = gamma
        self.var = t
    
    def length(self, interval:tuple, space):
        """
        Computes the length of the curve gamma from t0 to t1, in the metric G(gamma(t)):
        """
        t0, t1 = interval
        dgamma_dt = self.gamma.diff(self.var)
        Ggamma = space.metricG.subs({space.X[i]: self.gamma[i] for i in range(space.dim)})

        integrand = sp.sqrt((dgamma_dt.T * Ggamma * dgamma_dt)[0])
        return sp.integrate(integrand, (self.var, t0, t1))
    

