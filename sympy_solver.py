import sympy as sp


def solve_kepler(M, e):
    """ Solve Kepler's equation numerically with sympy's solver """
    E = sp.Symbol('E')
    zero_sympy = sp.nsolve(E - e * sp.sin(E) - M, E, 1)
    return float(zero_sympy)
