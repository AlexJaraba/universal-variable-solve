import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm
from tqdm import tqdm
from univ_prop.api.propagator import UniversalPropagator
from typing import Literal

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting.static import StaticOrbitPlotter

## Need to use specific enviroment to run this script

def find_mean_true(e_list,r_p,mu,dt,time):
    """
    Computes mean and true anomalies over time for elliptical, parabolic, or hyperbolic orbits.

    :param e_list: List of eccentricities to analyze.
    :type e_list: list[float]
    :param r_p: Radius at periapsis [km].
    :type r_p: float
    :param mu: Gravitational parameter [km³/s²].
    :type mu: float
    :param dt: Time step [s].
    :type dt: float
    :param time: Total duration to propagate [s].
    :type time: float

    :return: Tuple of (mean_anomalies, true_anomalies), both in degrees.
    :rtype: tuple[list[float], list[float]]
    """
    for e0 in e_list:
        if e0 < 1: # Elliptical Orbit
            a = r_p / (1-e0)
            v_p = np.sqrt(mu * ((2/r_p)-(1/a)))
            total_time = 2*np.pi * np.sqrt(a**3/mu)

            r_vec = np.array([r_p, 0, 0])
            v_vec = np.array([0, v_p, 0])

            steps = int(total_time // dt)
            true_anomalies = np.empty(steps)
            mean_anomalies = np.empty(steps)

            for i in range(steps):
                try:
                    r_vec, v_vec = UniversalPropagator.propagate(r_vec, v_vec, mu, dt)
                except RuntimeError:
                    continue

                r = norm(r_vec)
                h_vec = np.cross(r_vec, v_vec)
                h = norm(h_vec)
                e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
                e = norm(e_vec)

                # Compute true anomaly
                cos_true = np.dot(e_vec, r_vec) / (e * r)
                sin_true = np.dot(np.cross(e_vec, r_vec), h_vec) / (e * r * h)
                true_anomaly = np.arctan2(sin_true, cos_true)
                true_anomalies[i] = true_anomaly * 180 / np.pi

                E = 2 * np.arctan(np.tan(true_anomaly / 2) * np.sqrt(abs((1 - e)/(1 + e))))
                M = E - e * np.sin(E)
                mean_anomalies[i] = np.degrees(M)

        else:
            if np.isclose(e0, 1):  # Parabolic orbit
                a = r_p
                v_p = np.sqrt(2 * mu/r_p)
            else:  # Hyperbolic orbit
                a = -r_p / (1-e0)
                v_p = np.sqrt(mu * (2/r_p + 1/a))


            r_vec_f = np.array([r_p, 0, 0])
            v_vec_f = np.array([0, v_p, 0])
            r_vec_b = r_vec_f.copy()
            v_vec_b = v_vec_f.copy()

            steps = int(time // dt)
            true_anomalies = []
            mean_anomalies = []

            for _ in range(steps):
                try:
                    r_vec_f, v_vec_f = UniversalPropagator.propagate(r_vec_f, v_vec_f, mu, dt)
                except RuntimeError:
                    continue

                r = norm(r_vec_f)
                h_vec = np.cross(r_vec_f, v_vec_f)
                h = norm(h_vec)
                e_vec = np.cross(v_vec_f, h_vec) / mu - r_vec_f / r
                e = norm(e_vec)

                # Compute true anomaly
                cos_true = np.dot(e_vec, r_vec_f) / (e * r)
                sin_true = np.dot(np.cross(e_vec, r_vec_f), h_vec) / (e * r * h)
                true_anomaly = np.arctan2(sin_true, cos_true)
                true_anomalies.append(np.degrees(true_anomaly))

                if np.isclose(e0, 1):  # Parabolic case
                    M = 0.5 * np.tan(true_anomaly/2) + (1/6) * (np.tan(true_anomaly/2))**3

                else:  # Hyperbolic case
                    num = np.sqrt(e + 1) + np.sqrt(e - 1) * np.tan(true_anomaly/2)
                    den = np.sqrt(e + 1) - np.sqrt(e - 1) * np.tan(true_anomaly/2)
                    F = np.log(num/den)
                    M = e * np.sinh(F) - F
                
                mean_anomalies.append(M)
            
            for _ in range(steps):
                try:
                    r_vec_b, v_vec_b = UniversalPropagator.propagate(r_vec_b, v_vec_b, mu, -dt)
                except RuntimeError:
                    continue

                r = norm(r_vec_b)
                h_vec = np.cross(r_vec_b, v_vec_b)
                h = norm(h_vec)
                e_vec = np.cross(v_vec_b, h_vec) / mu - r_vec_b / r
                e = norm(e_vec)

                # Compute true anomaly
                cos_true = np.dot(e_vec, r_vec_b) / (e * r)
                sin_true = np.dot(np.cross(e_vec, r_vec_b), h_vec) / (e * r * h)
                true_anomaly = np.arctan2(sin_true, cos_true)
                true_anomalies.append(np.degrees(true_anomaly))

                if np.isclose(e0, 1):  # Parabolic case
                    M = 0.5 * np.tan(true_anomaly/2) + (1/6) * (np.tan(true_anomaly/2))**3

                else:  # Hyperbolic case
                    num = np.sqrt(e + 1) + np.sqrt(e - 1) * np.tan(true_anomaly/2)
                    den = np.sqrt(e + 1) - np.sqrt(e - 1) * np.tan(true_anomaly/2)
                    F = np.log(num/den)
                    M = e * np.sinh(F) - F
                
                mean_anomalies.append(M)

    return mean_anomalies, true_anomalies

def find_orbit(r_p, e0, mu, total_time, dt):
    """
    Numerically propagate a 2-body orbit using a universal variable solver.

    :param r_p: Radius at periapsis [km].
    :type r_p: float
    :param e0: Orbital eccentricity.
    :type e0: float
    :param mu: Gravitational parameter [km³/s²].
    :type mu: float
    :param total_time: Total time to propagate [s].
    :type total_time: float
    :param dt: Time step for integration [s].
    :type dt: float

    :return: Tuple of propagated position array, velocity array, and time array.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    if np.isclose(e0, 1):  # Parabolic orbit
        v_p = np.sqrt(2 * mu/r_p)
    elif e0 > 1:
        a = -r_p / (1-e0)
        v_p = np.sqrt(mu * ((2 / r_p) + (1 / a)))
    else:
        a = r_p / (1 - e0)
        v_p = np.sqrt(mu * ((2 / r_p) - (1 / a)))

    r_vec0 = np.array([r_p, 0, 0])
    v_vec0 = np.array([0, v_p, 0])

    if np.isclose(e0, 1.0):
        time_span = total_time / 2  # shorter time span
    else:
        time_span = total_time

    steps = int(time_span // dt)
    r_arr = np.empty((steps, 3))
    v_arr = np.empty((steps, 3))
    time = np.empty(steps)

    r_vec = r_vec0.copy()
    v_vec = v_vec0.copy()

    for i in range(steps):
        t = i * dt
        time[i] = t
        try:
            r_vec, v_vec = UniversalPropagator.propagate(r_vec, v_vec, mu, dt)  # propagate by dt incrementally
            r_arr[i] = r_vec
            v_arr[i] = v_vec
        except RuntimeError:
            r_arr[i] = np.array([np.nan, np.nan, np.nan])
            v_arr[i] = np.array([np.nan, np.nan, np.nan])

    return r_arr, v_arr, time


def main(
    Choice_of_test: Literal[1, 2, 3, 4],
    dt: float,
    radius_at_periapsis: float,
    mu: float,
    total_time: float,
    eccentricity_list: list[float]
) -> None:
    """
    Run orbital propagation visualizations and error analysis using custom and Poliastro propagators.

    :param Choice_of_test: Test case selector:
                           - 1: Plot orbits using the custom propagator
                           - 2: Plot orbits using Poliastro
                           - 3: Compare final position errors vs eccentricity
                           - 4: Compare true and mean anomaly errors over time
    :type Choice_of_test: Literal[1, 2, 3, 4]
    :param dt: Propagation time step in seconds.
    :type dt: float
    :param radius_at_periapsis: Radius at periapsis in kilometers.
    :type radius_at_periapsis: float
    :param mu: Standard gravitational parameter in km³/s².
    :type mu: float
    :param total_time: Total duration of propagation in seconds.
    :type total_time: float
    :param eccentricity_list: List of eccentricities to simulate.
    :type eccentricity_list: list[float]

    :return: None
    :rtype: None
    """
    start_time = time.time()

    # ---- parameters ----
    r_p = radius_at_periapsis           # periapsis radius [km]
    time_total = total_time    # propagation time [s]
    e_list = eccentricity_list

    # Pre‑define the fixed angles
    inc  = 0.0 * u.deg
    raan = 0.0 * u.deg
    argp = 0.0 * u.deg
    nu   = 0.0 * u.deg

    d = Choice_of_test
    if d not in {1, 2, 3, 4}:
        raise ValueError(f"Choice_of_test must be one of [1, 2, 3, 4], got {d}")

    if d == 1:
        # --- 1) Plot your custom orbits ---
        plt.figure(figsize=(6,6))
        for e0 in e_list:
            try:
                r_arr, v_arr, t_arr = find_orbit(r_p, e0, mu, time_total, dt)
                plt.plot(r_arr[:,0], r_arr[:,1], label=f"e={e0:.2f}")
            except Exception as ex:
                print(f"[Custom] e={e0}: {ex}")
        plt.plot(0,0,'yo', label="Earth")
        plt.axis("equal"); plt.grid(True)
        plt.title("Custom Propagator Orbits")
        plt.xlabel("x [km]"); plt.ylabel("y [km]")
        plt.legend()

    if d == 2:
        # --- 2) Plot Poliastro orbits on the same axes ---
        fig, ax = plt.subplots(figsize=(6,6))
        plotter = StaticOrbitPlotter(ax)
        for e0 in e_list:
            label = f"e={e0:.2f}"
            if np.isclose(e0, 1.0):
                # parabolic
                p = 2 * r_p * u.km
                orb = Orbit.parabolic(Earth, p, inc, raan, argp, nu)
            else:
                # elliptical or hyperbolic
                a = (r_p / (1 - e0)) if e0 < 1 else (-r_p / (e0 - 1))
                orb = Orbit.from_classical(
                    Earth,
                    a * u.km,
                    e0 * u.one,
                    inc, raan, argp, nu
                )
            plotter.plot(orb, label=label)
        ax.set_title("Poliastro Orbits"); ax.grid(True)

    if d == 3:
        # --- 3) Compute error at final time and plot vs e ---
        errors = []
        for e0 in e_list:
            try:
                # custom final position
                r_arr, _, _ = find_orbit(r_p, e0, mu, time_total, dt)
                r_custom = r_arr[-1]

                # poliastro at same final time
                if np.isclose(e0, 1.0):
                    orb = Orbit.parabolic(Earth, 2*r_p*u.km, inc, raan, argp, nu)
                else:
                    a = (r_p / (1 - e0)) if e0 < 1 else (-r_p / (e0 - 1))
                    orb = Orbit.from_classical(
                        Earth,
                        a * u.km,
                        e0 * u.one,
                        inc, raan, argp, nu
                    )
                # sample at exactly time_total
                sampled = orb.propagate(time_total * u.s)
                r_poliastro = sampled.r.to_value(u.km)

                errors.append(np.linalg.norm(r_custom - r_poliastro))
            except Exception as ex:
                print(f"[Error] e={e0}: {ex}")
                errors.append(np.nan)

        # plot error vs eccentricity
        plt.figure(figsize=(6,4))
        plt.semilogy(e_list, errors, 'o-')
        plt.xlabel("Eccentricity")
        plt.ylabel("Final Position Error [km]")
        plt.title("Custom vs Poliastro Error")
        plt.grid(True)

        plt.show()
        print(f"Elapsed time: {time.time() - start_time:.2f} s")

    if d == 4:
        # --- 4) Compute mean and true anomaly errors for one representative elliptical orbit ---
        # Choose an example eccentricity
        e0 = 0.5
        r_arr, v_arr, t_arr = find_orbit(r_p, e0, mu, time_total, dt)

        # Initialize Poliastro orbit at t=0
        if np.isclose(e0, 1.0): # parabolic
            p = 2 * r_p * u.km
            orb = Orbit.parabolic(Earth, p, inc, raan, argp, nu)
        else: # elliptical or hyperbolic
            a = (r_p / (1 - e0)) if e0 < 1 else (-r_p / (e0 - 1))
            orb = Orbit.from_classical(Earth,a * u.km,e0 * u.one,inc, raan, argp, nu)
        
        true_anomaly_errors = []
        mean_anomaly_errors = []

        for i, t in enumerate(t_arr):
            r_vec = r_arr[i]
            v_vec = v_arr[i]

            # Skip if nan values (failed propagation)
            if np.isnan(r_vec).any() or np.isnan(v_vec).any():
                continue
            
            M_mine, nu_mine = find_mean_true(e_list,r_p,mu,dt,time_total)

            # Poliastro propagated orbit
            orb_t = orb.propagate(t * u.s)
            nu_poliastro = orb_t.nu.to(u.deg).value
            M_poliastro = orb_t.M.to(u.deg).value

            # Calculate absolute difference in degrees (mod 360)
            d_nu = abs((nu_mine - nu_poliastro + 180) % 360 - 180)
            d_M = abs((M_mine - M_poliastro + 180) % 360 - 180)

            true_anomaly_errors.append(d_nu)
            mean_anomaly_errors.append(d_M)

        plt.figure(figsize=(8,5))
        plt.plot(t_arr[:len(true_anomaly_errors)], true_anomaly_errors, label="True Anomaly Error [deg]")
        plt.plot(t_arr[:len(mean_anomaly_errors)], mean_anomaly_errors, label="Mean Anomaly Error [deg]")
        plt.xlabel("Time [s]")
        plt.ylabel("Absolute Error [deg]")
        plt.title(f"Anomaly Errors for e = {e0}")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Elapsed time: {time.time() - start_time:.2f} s")
