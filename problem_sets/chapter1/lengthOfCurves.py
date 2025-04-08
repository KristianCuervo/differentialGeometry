import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from differentialgeometry import Surface, Curve, Space, Map
import sympy as sp

"""
Exercise 1.29
"""
x1, x2 = sp.symbols('x1 x2')

# surface defined
surfaceR = Surface(expr=sp.Matrix([x1, x2 , x1**2 + x2**2]), var=sp.Matrix([x1, x2]))
print(surfaceR.metricG) # a) Gu is correct


# Diffeomorphism phi
phiMap = Map(expr=sp.Matrix([sp.sqrt(x1**2 + x2**2), sp.arg(x1 + x2*sp.I)]),
             var=sp.Matrix([x1, x2]))
metricGv = phiMap.metricGv(surfaceR)
print(metricGv)


# Curves
interval = (-sp.pi, +sp.pi)
t = sp.symbols('t')

gamma = sp.Matrix([sp.cos(t), sp.sin(t)])
eta = sp.Matrix([1, t])


