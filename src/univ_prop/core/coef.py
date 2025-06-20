import numpy as np
from numpy.linalg import norm
from .stumpff import *

def f(chi,r0,C):
    return 1 - (chi**2 / r0) * C

def g(dt,chi,mu,S):
    return dt - (1 / np.sqrt(mu)) * chi**3 * S

def df(chi,r0,r,S,z,mu):
    return (np.sqrt(mu) / (r * r0)) * chi * (z * S - 1)

def dg(chi,r,C):
    return 1 - (chi**2 / r) * C
