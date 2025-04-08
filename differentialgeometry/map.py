import sympy as sp
from .space import Space

class Map:
    def __init__(self, expr:sp.Matrix, domain_vars:sp.Matrix, codomain_vars:sp.Matrix):
        
        # (codomain_vars) = expr(domain_vars)
        self.expr = expr 
        self.domain_vars = domain_vars
        self.codomain_vars = codomain_vars

        self.isDiffeomorphism = self.checkDiffeomorphism()
    
    def checkDiffeomorphism(self):
        try:
            # Check if the Jacobian is invertible
            inverse = self.j.inv()
            if inverse is not None:
                return True
            else:
                return False
            
        except:
            return False
    
    def getInverseMapping(self):
        """
        Returns the inverse mapping of the map phi, and associated variables.
        Creates a new mapping - not saved to the object.
        """
        if not self.isDiffeomorphism:
            raise ValueError("Map is not a diffeomorphism; cannot compute inverse.")
        
        # Create symbolic variables y1, y2, ..., yn
        Y = sp.Matrix(sp.symbols(f'y1:{self.dim+1}'))  # y1:dim+1 creates y1, y2, ..., yN

        # Create equations: y_i = f_i(x)
        equations = [sp.Eq(Y[i], self.expr[i]) for i in range(self.dim)]

        # Solve the system for X in terms of Y
        sol = sp.solve(equations, self.var, dict=True)

        if not sol:
            raise ValueError("Could not symbolically solve for inverse.")
        
        # Return the first solution (sp.solve can return multiple)
        inverseMap = sp.Matrix([sol[0][var] for var in self.var])
        return Map(expr=inverseMap, var=Y)
    
    def metricGv(self, space):
        """
        Pullback formula: this returns the new metric which the mapping creates
        from the spaces U --> V. 
        """
        
        JphiInv = self.inv.jacobian(self.invVar)
        Gu_y = space.metricG.subs({space.X[i]: self.invVar[i] for i in range(self.dim)})
        return JphiInv * Gu_y * JphiInv.T

    def fromUtoV(self, space):
        metricGu_inY = space.metricG.subs({space.var[i]: self.var[i] for i in range(self.dim)})
        metricGv = self.j * metricGu_inY * self.j.T

        return Space(G=metricGv)