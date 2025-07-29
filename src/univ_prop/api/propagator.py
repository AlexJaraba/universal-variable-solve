import numpy as np
from numpy.linalg import norm
from ..pyuniversal.kepler import solve_chi
from ..pyuniversal.coef import f, g, df, dg
from ..pyuniversal.stumpff import stumpff_C, stumpff_S
from typing import Tuple

class UniversalPropagator:
    def __init__(self, mu: float, name: str):
        self.name = name
        self.mu = mu
        self.tol = 1e-8
    
    def propagate(
        self,
        Position: list[float] | np.ndarray,
        Velocity: list[float] | np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate the orbit forward by one time step using universal variable formulation.

        :param Position: Initial position vector [x, y, z] in kilometers.
        :type Position: list[float] or numpy.ndarray
        :param Velocity: Initial velocity vector [vx, vy, vz] in km/s.
        :type Velocity: list[float] or numpy.ndarray
        :param dt: Time step duration in seconds.
        :type dt: float

        :return: A tuple containing:
                 - r_vec (numpy.ndarray): Propagated position vector in km.
                 - v_vec (numpy.ndarray): Propagated velocity vector in km/s.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """

        mu = self.mu
        tol = self.tol

        r0_vec = np.array(Position)
        v0_vec = np.array(Velocity)
        r0 = norm(r0_vec)
        v0 = norm(v0_vec)
        vr = np.dot(r0_vec, v0_vec) / r0

        alpha = 2 / r0 - v0**2 / mu

        try:
            chi = solve_chi(mu, alpha, r0_vec, vr, dt, tol=tol, verbose=False)
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