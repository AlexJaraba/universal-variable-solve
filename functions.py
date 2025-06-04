import numpy as np
from math import sqrt, sin, cos, cosh, sinh, log

def stumpff_C(z):
    if z > 0:
        return (1 - cos(sqrt(z)))/z
    elif z < 0:
        return (cosh(sqrt(-z)-1))/(-z)
    else:
        return 0.5

def stumpff_S(z):
    if z > 0:
        return (sqrt(z)-sin(sqrt(z)))/(sqrt(z))**3
    elif z < 0:
        return (sinh(sqrt(-z))-sqrt(-z))(sqrt(z))**3
    else:
        return 1/6

def uniKepler_F(chi,mu,alpha,r_0,vr_0,dt):
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    s_mu = sqrt(mu)
    return ((r_0 * vr_0) / s_mu) * (chi**2 * C(z)) + ((1 - alpha * r_0) * chi**3 * S(z)) + (r_0 * chi) - (s_mu *dt)

def uniKepler_dFdchi(chi,mu,alpha,r_0,vr_0):
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    s_mu = sqrt(mu)
    return (((r_0*vr_0)/s_mu)*chi*(1-(z*S))) + ((1-alpha*r_0)*chi**2*C) + r_0

def newton_solver(mu,alpha,r_0,vr_0,dt,steps):

    tol = 1e-8

    if alpha > 0:
        chi = sqrt(mu) * dt * alpha
    elif alpha < 0:
        chi = sqrt(-1/alpha) * log((-2*mu*alpha*dt) / (r_0*vr_0 + sqrt(-mu/alpha)*(1 - r_0*alpha)))
    else:
        chi = sqrt(2 * mu) * dt / r_0

    for _ in range(steps):
        f = uniKepler_F(chi,mu,alpha,r_0,vr_0,dt)
        df = uniKepler_dFdchi(chi,mu,alpha,r_0,vr_0)
        dchi = -f/df
        chi += dchi
        if abs(dchi) < tol:
            break
    else:
        raise RuntimeError("Newton-Raphson did not converge")
    
    return chi

def propagate(r0_vec,v0_vec,mu,dt,steps):
    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    vr = np.dot(r0,v0) / r0

    alpha = 2/r0 -v0**2/mu

    chi = newton_solver(mu,alpha,r0,vr,dt,steps)

    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)

    f = 1 - (chi**2/r0) * C
    g = dt - (1/sqrt(mu)) * chi**3 * S

    r_vec = f * r0_vec + g * v0_vec
    r = np.linalg.norm(r_vec)

    fdot = (sqrt(mu) / (r * r0)) * chi * (z * S - 1)
    gdot = 1 - (chi**2 / r) * C

    v_vec = fdot * r0_vec + gdot * v0_vec

    return r_vec, v_vec