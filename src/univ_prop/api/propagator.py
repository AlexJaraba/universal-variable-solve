import numpy as np
from numpy.linalg import norm
from core import Chi_newton, stumpff_C, stumpff_S, f, g, df, dg

def propagate(r0_vec, v0_vec, mu, dt, tol=1e-8):
    r0_vec = np.array(r0_vec)
    v0_vec = np.array(v0_vec)
    r0 = norm(r0_vec)
    v0 = norm(v0_vec)
    vr = np.dot(r0_vec, v0_vec) / r0

    alpha = 2 / r0 - v0**2 / mu

    try:
        chi = Chi_newton(mu, alpha, r0_vec, vr, dt, tol=tol, verbose=False)
    except RuntimeError as e:
        print("Propagation failed:", e)
        raise

    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)

    _f = f(chi,r0,C)
    _g = g(dt,chi,mu,S)

    r_vec = _f * r0_vec + _g * v0_vec
    r = norm(r_vec)

    fdot = df(chi,r0,r,S,z,mu)
    gdot = dg(chi,r,C)

    v_vec = fdot * r0_vec + gdot * v0_vec
    
    return r_vec, v_vec