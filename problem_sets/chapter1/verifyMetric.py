import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import differentialgeometry as dg
import sympy as sp

"""
Exercise 1.23
"""

x1, x2 = sp.symbols('x1 x2')
X = sp.Matrix([x1, x2])
G_given = sp.Matrix([[4*x1**2 + 1, 4*x1*x2],
                     [4*x1*x2, 4*x2**2 + 1]])
print("G_given:")
print(G_given)

S = dg.Surface(sp.Matrix([x1, x2, x1**2 + x2**2]), X)
G_computed = S.G
print("G_computed:")
print(G_computed)

if G_given.equals(G_computed):
    print("The metric tensors are equal.")