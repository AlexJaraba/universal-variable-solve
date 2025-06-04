import numpy as np
from math import sqrt, sin, cos, cosh, sinh

def C(z):
    if z > 0:
        return (1 - cos(sqrt(z)))/z
    elif z < 0:
        return (cosh(sqrt(-z)-1))/(-z)
    else:
        return 0.5

def S(z):
    if z > 0:
        return (sqrt(z)-sin(sqrt(z)))/(sqrt(z))**3
    elif z < 0:
        return (sinh(sqrt(-z))-sqrt(-z))(sqrt(z))**3
    else:
        return 1/6