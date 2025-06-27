import numpy as np
import pytest
from univ_prop.api.propagator import UniversalPropagator

def test_propagate_returns_correct_shapes(mu,r0,v0):
    prop = UniversalPropagator(mu)

    r0 = np.array(r0)
    v0 = np.array(v0)

    r_new, v_new = prop.propagate(r0, v0)

    assert isinstance(r_new, np.ndarray), "r_new should be a numpy array"
    assert isinstance(v_new, np.ndarray), "v_new should be a numpy array"
    assert r_new.shape == (3,), "r_new should be a 3D vector"
    assert v_new.shape == (3,), "v_new should be a 3D vector"

def test_propagate_position_changes(mu,r0,v0):
    prop = UniversalPropagator(mu)

    r0 = np.array(r0)
    v0 = np.array(v0)

    r_new, _ = prop.propagate(r0, v0)

    assert not np.allclose(r0, r_new), "Position vector should change after propagation"

def test_propagate_velocity_changes(mu,r0,v0):
    prop = UniversalPropagator(mu)

    r0 = np.array(r0)
    v0 = np.array(v0)

    _, v_new = prop.propagate(r0, v0)

    assert not np.allclose(v0, v_new), "Velocity vector should change after propagation"