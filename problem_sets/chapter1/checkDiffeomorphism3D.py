import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import differentialgeometry as dg
import sympy as sp

"""
Exercise 1.15
"""
x1, x2, x3 = sp.symbols('x1 x2 x3')
X = sp.Matrix([x1, x2, x3])
phi = dg.Map(sp.Matrix([x2, x3, x1+x2+x3]), X)

# a) show that phi is a diffeomorphism
print(phi.isDiffeomorphism)

# b) find Jphi and Jphi^-1
print("Jphi = ", phi.j)
phiInv = dg.Map(phi.inverseMapping()[0],
                phi.inverseMapping()[1])
print("Jphi^-1 = ", phiInv.j)

# c) Show that these jacobian matrices are the inverses of each other
print("J^-1 phi = ", phi.j.inv())
print("JphiInv^in X", phiInv.j.subs({phi.inverseMapping()[1][0]: phi.expr[0],
                                      phi.inverseMapping()[1][1]: phi.expr[1],
                                      phi.inverseMapping()[1][2]: phi.expr[2]}))

if phi.j.inv() == phiInv.j.subs({phi.inverseMapping()[1][0]: phi.expr[0],
                                      phi.inverseMapping()[1][1]: phi.expr[1],
                                      phi.inverseMapping()[1][2]: phi.expr[2]}):
    print("The jacobians are inverses of each other")
