import sympy as sp

class Map:
    def __init__(self, phi:sp.Matrix, X:sp.Matrix):
        self.f = phi
        self.X = X
        self.dim = len(X)
        self.j = self.f.jacobian(self.X)
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
    
    def inverseMapping(self):
        """
        Returns the inverse mapping of the map phi, and associated variables.
        Creates a new mapping - not saved to the object.
        """
        if not self.isDiffeomorphism:
            raise ValueError("Map is not a diffeomorphism; cannot compute inverse.")
        
        # Create symbolic variables y1, y2, ..., yn
        Y = sp.Matrix(sp.symbols(f'y1:{self.dim+1}'))  # y1:dim+1 creates y1, y2, ..., yN

        # Create equations: y_i = f_i(x)
        equations = [sp.Eq(Y[i], self.f[i]) for i in range(self.dim)]

        # Solve the system for X in terms of Y
        sol = sp.solve(equations, self.X, dict=True)

        if not sol:
            raise ValueError("Could not symbolically solve for inverse.")
        
        # Return the first solution (sp.solve can return multiple)
        inverseMap = sp.Matrix([sol[0][var] for var in self.X])
        return inverseMap, Y