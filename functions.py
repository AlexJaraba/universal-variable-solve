import numpy as np
import matplotlib.pyplot as plt
import time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from numpy.linalg import norm

# --- Stumpff Functions ---
def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / -z
    else:
        return 0.5

def stumpff_S(z):
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3)
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)**3)
    else:
        return 1 / 6

# --- Universal Kepler's Equation and its Derivative ---
def uniKepler_F(chi, mu, alpha, r_0, vr_0, dt):
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    sqrt_mu = np.sqrt(mu)
    return ((r_0 * vr_0) / sqrt_mu) * (chi**2 * C) + ((1 - alpha * r_0) * chi**3 * S) + (r_0 * chi) - (sqrt_mu * dt)

def uniKepler_dFdchi(chi, mu, alpha, r_0, vr_0):
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    sqrt_mu = np.sqrt(mu)
    return ((r_0 * vr_0) / sqrt_mu) * chi * (1 - z * S) + (1 - alpha * r_0) * chi**2 * C + r_0

# --- Newton Solver ---
def Newton_Solver(tol, function, d_function, x0, max_iter=50):
    x = x0
    for _ in range(max_iter):
        f = function(x)
        df = d_function(x)
        if df == 0 or not np.isfinite(df):
            break
        dx = -f / df
        if not np.isfinite(dx):
            break
        x += dx
        if abs(dx) < tol:
            return x
    raise RuntimeError("Newton Solver did not converge")

# --- Chi Solver ---
def Chi_newton(mu, alpha, r_0, vr_0, dt):
    tol = 1e-8
    if alpha > 0:
        chi = np.sqrt(mu) * dt * abs(alpha)
    elif alpha < 0:
        arg = (-2 * mu * alpha * dt) / (r_0 * vr_0 + np.sqrt(-mu / alpha) * (1 - r_0 * alpha))
        chi = np.sqrt(-1 / alpha) * dt if arg <= 0 else np.sign(dt) * np.sqrt(-1 / alpha) * np.log(arg)
    else:
        chi = np.sqrt(2 * mu) * dt / r_0
    return Newton_Solver(tol,
                         lambda chi: uniKepler_F(chi, mu, alpha, r_0, vr_0, dt),
                         lambda chi: uniKepler_dFdchi(chi, mu, alpha, r_0, vr_0),
                         chi)

# --- Orbit Propagation ---
def propagate(r0_vec, v0_vec, mu, dt):
    r0_vec = np.array(r0_vec)
    v0_vec = np.array(v0_vec)
    r0 = norm(r0_vec)
    v0 = norm(v0_vec)
    vr = np.dot(r0_vec, v0_vec) / r0
    alpha = 2 / r0 - v0**2 / mu
    chi = Chi_newton(mu, alpha, r0, vr, dt)
    z = alpha * chi**2
    C = stumpff_C(z)
    S = stumpff_S(z)
    f = 1 - (chi**2 / r0) * C
    g = dt - (1 / np.sqrt(mu)) * chi**3 * S
    r_vec = f * r0_vec + g * v0_vec
    r = norm(r_vec)
    fdot = (np.sqrt(mu) / (r * r0)) * chi * (z * S - 1)
    gdot = 1 - (chi**2 / r) * C
    v_vec = fdot * r0_vec + gdot * v0_vec
    return r_vec, v_vec

def plot_mean_true(e_list,r_p,mu,dt,time):
    for e in e_list:
        if e < 1:
            a = r_p / (1-e)
            v_p = np.sqrt(mu * ((2/r_p)-(1/a)))
            total_time = 2*np.pi * np.sqrt(a**3/mu)
        elif e == 1:  # Parabolic orbit
            v_p = np.sqrt(2 * mu/r_p)
            total_time = time
        else:  # Hyperbolic orbit
            a = -r_p / (1-e)  # negative semi-major axis for hyperbola
            v_p = np.sqrt(mu * (2/r_p + 1/a))
            total_time = time
        
        r_vec = np.array([r_p, 0, 0])
        v_vec = np.array([0, v_p, 0])

        steps = int(total_time // dt)
        true_anomalies = np.empty(steps)
        mean_anomalies = np.empty(steps)

        for i in range(steps):
            r_vec, v_vec = propagate(r_vec, v_vec, mu, dt)
            r = norm(r_vec)
            v = norm(v_vec)
            h_vec = np.cross(r_vec, v_vec)
            h = norm(h_vec)
            e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
            ecc = norm(e_vec)
            t = (i + 1) * dt

            # Calculate anomalies
            if ecc < 1:
                T = (2 * np.pi / np.sqrt(mu)) * a**(3/2)
                M_e = 2 * np.pi * t / T
                nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (ecc * r), -1, 1))
                if np.dot(r_vec, v_vec) < 0:
                    nu = 2 * np.pi - nu
                mean_anomalies[i] = np.degrees(M_e)
                true_anomalies[i] = np.degrees(nu)
            elif ecc > 1:
                M_h = (mu**2 / h**3) * (ecc**2 - 1)**(3 / 2) * t
                F0 = np.log(2 * M_h / ecc + 1.8)
                F = Newton_Solver(1e-8,
                                  lambda F_: ecc * np.sinh(F_) - F_ - M_h,
                                  lambda F_: ecc * np.cosh(F_) - 1,
                                  F0)
                nu = 2 * np.arctan(np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(F / 2))
                mean_anomalies[i] = np.radians(M_h)
                true_anomalies[i] = np.degrees(nu)
            else:
                M_p = mu**2 / h**3 * t
                z = (3 * M_p + np.sqrt(1 + (3 * M_p)**2))**(1 / 3)
                nu = 2 * np.arctan(z - 1 / z)
                mean_anomalies[i] = np.radians(M_p)
                true_anomalies[i] = np.degrees(nu)

        true_anomalies = ((np.unwrap(true_anomalies) + 180) % 360) - 180
        mean_anomalies = ((np.unwrap(mean_anomalies) + 180) % 360) - 180
        plt.scatter(true_anomalies,mean_anomalies, s=1, label=f"e={e}")
        
    plt.legend()

# --- Main Plotting Routine ---
if __name__ == "__main__":
    start_time = time.time()

    dt = 10  # time step in seconds
    e_list_elpic = [1e-8, 0.5, 0.99]
    e_list_para = [1]
    e_list_hyper = [1.2, 1.5, 2.0, 5, 12]
    r_p = 10000  # km
    mu = 398600.4418  # km^3/s^2
    time_total = 5400 * 8  # seconds

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_mean_true(e_list_elpic, r_p, mu, dt, time_total)
    plt.title("Elliptical Orbits")
    plt.xlabel("True Anomaly (deg)")
    plt.ylabel("Mean Anomaly (deg)")
    plt.grid(True)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    plt.subplot(1, 3, 2)
    plot_mean_true(e_list_para, r_p, mu, dt, time_total)
    plt.title("Parabolic Orbit")
    plt.xlabel("True Anomaly (deg)")
    plt.ylabel("Mean Anomaly (rad)")
    plt.grid(True)
    plt.xlim(-180, 180)
    plt.ylim(-np.pi, np.pi)

    plt.subplot(1, 3, 3)
    plot_mean_true(e_list_hyper, r_p, mu, dt, time_total)
    plt.title("Hyperbolic Orbits")
    plt.xlabel("True Anomaly (deg)")
    plt.ylabel("Mean Anomaly (rad)")
    plt.grid(True)
    plt.xlim(-180, 180)
    plt.ylim(-np.pi, np.pi)

    plt.tight_layout()
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")