import sympy as sp
import numpy as np
from .manifold import RiemannianManifold
from .utils import compute_christoffel_symbols, compute_riemann_tensor

class Space(RiemannianManifold):
    def __init__(self, metric:sp.Matrix, coord_vars:sp.Matrix):
        super().__init__(vars=coord_vars, metric=metric)
