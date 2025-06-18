import numpy as np
from numpy.linalg import norm
from Newton_Solve import Newton_Solver
from stumpff import *

def Chi_newton(mu, alpha, r0, vr, dt, tol=1e-8, max_iter=100, verbose=False):
    chi0 = np.sqrt(mu) * dt * abs(alpha)
    r = norm(r0)

    def F(chi):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        return (r * vr / np.sqrt(mu)) * chi**2 * C + (1 - r * alpha) * chi**3 * S + r * chi - np.sqrt(mu) * dt
    
    def dF(chi):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        return (r * vr / np.sqrt(mu)) * chi * (1 - z * S) + (1 - r * alpha) * chi**2 * C + r
    
    if verbose:
        print(f"alpha={alpha}, chi0={chi0}, z={alpha * chi0**2}")

    if not np.isfinite(chi0):
        raise ValueError("Inital guess chi0 is not finite")
    
    if np.abs(alpha * chi0**2) > 1e6:
        raise ValueError("Chi0 guess leads to extreme z value")

    return Newton_Solver(tol, F, dF, chi0, max_iter=max_iter, verbose=verbose)