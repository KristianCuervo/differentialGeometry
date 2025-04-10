import sympy as sp
import numpy as np

def compute_christoffel_symbols(metric: sp.Matrix, coord_vars: sp.Matrix):
    dim = len(coord_vars)
    Gamma = sp.zeros(dim, dim, dim)
    metric_inv = metric.inv()
    for i in range(dim):
        for j in range(dim):
            for m in range(dim):
                term = 0
                for l in range(dim):
                    term += 0.5 * (
                        metric[j, l].diff(coord_vars[i]) +
                        metric[l, i].diff(coord_vars[j]) -
                        metric[i, j].diff(coord_vars[l])
                    ) * metric_inv[m, l]
                Gamma[i, j, m] = sp.simplify(term)
    return Gamma

def compute_riemann_tensor(christoffels, coord_vars):
    dim = len(coord_vars)
    R = np.zeros((dim, dim, dim, dim), dtype=object)
    Γ = christoffels
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    term = Γ[j, k, m].diff(coord_vars[i]) - Γ[i, k, m].diff(coord_vars[j])
                    for s in range(dim):
                        term += Γ[j, k, s]*Γ[i, s, m] - Γ[i, k, s]*Γ[j, s, m]
                    R[i, j, k, m] = sp.simplify(term)
    return R