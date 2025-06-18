import numpy as np

def stumpff_C(z):
    if z > 0:
        sqrt_z = np.sqrt(z)
        return (1 - np.cos(sqrt_z)) / z
    elif np.isclose(z, 0):
        return 0.5
    else:
        sqrt_neg_z = np.sqrt(-z)
        if sqrt_neg_z > 50:
            return 0.0
        return (1 - np.cosh(sqrt_neg_z)) / -z
    
def stumpff_S(z):
    if z > 0:
        sqrt_z = np.sqrt(z)
        return ((sqrt_z - np.sin(sqrt_z))) / (sqrt_z**3)
    elif np.isclose(z, 0):
        return 1/6
    else:
        sqrt_neg_z = np.sqrt(-z)
        if sqrt_neg_z > 50:
            return 0.0
        return (np.sinh(sqrt_neg_z) - sqrt_neg_z) / (sqrt_neg_z**3)