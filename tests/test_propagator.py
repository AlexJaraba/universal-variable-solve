import numpy as np
import pytest
from univ_prop.api.propagator import UniversalPropagator

## Need to use specific environment to run this script

@pytest.mark.parametrize(
    "mu, r0, v0, dt",
    [
        (398600.4418, [7000, 0, 0], [0, 7.5, 0], 60),  # LEO-like orbit
        (398600.4418, [10000, 0, 0], [0, 5, 0], 120),  # Higher orbit
    ]
)
def test_propagate_returns_correct_shapes(mu, r0, v0, dt):
    """
    Test that the propagate method returns 3D vectors as NumPy arrays.

    :param mu: Gravitational parameter (km³/s²).
    :type mu: float
    :param r0: Initial position vector.
    :type r0: list or np.ndarray
    :param v0: Initial velocity vector.
    :type v0: list or np.ndarray
    :param dt: Time step (seconds).
    :type dt: float
    """
    prop = UniversalPropagator(mu, name="test")
    r_new, v_new = prop.propagate(np.array(r0), np.array(v0), dt)

    assert isinstance(r_new, np.ndarray), "r_new should be a numpy array"
    assert isinstance(v_new, np.ndarray), "v_new should be a numpy array"
    assert r_new.shape == (3,), "r_new should be a 3D vector"
    assert v_new.shape == (3,), "v_new should be a 3D vector"

@pytest.mark.parametrize(
    "mu, r0, v0, dt",
    [
        (398600.4418, [7000, 0, 0], [0, 7.5, 0], 60),
    ]
)
def test_propagate_position_changes(mu, r0, v0, dt):
    """
    Test that the position vector changes after propagation.

    :param mu: Gravitational parameter (km³/s²).
    :type mu: float
    :param r0: Initial position vector.
    :type r0: list or np.ndarray
    :param v0: Initial velocity vector.
    :type v0: list or np.ndarray
    :param dt: Time step (seconds).
    :type dt: float
    """
    prop = UniversalPropagator(mu, name="test")
    r0 = np.array(r0)
    v0 = np.array(v0)
    r_new, _ = prop.propagate(r0, v0, dt)

    assert not np.allclose(r0, r_new), "Position should change after propagation"

@pytest.mark.parametrize(
    "mu, r0, v0, dt",
    [
        (398600.4418, [7000, 0, 0], [0, 7.5, 0], 60),
    ]
)
def test_propagate_velocity_changes(mu, r0, v0, dt):
    """
    Test that the velocity vector changes after propagation.

    :param mu: Gravitational parameter (km³/s²).
    :type mu: float
    :param r0: Initial position vector.
    :type r0: list or np.ndarray
    :param v0: Initial velocity vector.
    :type v0: list or np.ndarray
    :param dt: Time step (seconds).
    :type dt: float
    """
    prop = UniversalPropagator(mu, name="test")
    r0 = np.array(r0)
    v0 = np.array(v0)
    _, v_new = prop.propagate(r0, v0, dt)

    assert not np.allclose(v0, v_new), "Velocity should change after propagation"