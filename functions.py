import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm
from tqdm import tqdm

# --- Stumpff Functions ---
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

# --- Newton Solver ---
def Newton_Solver(tol, function, d_function, x0, max_iter=100, verbose=False):
    x = x0
    for i in range(max_iter):
        f = function(x)
        df = d_function(x)
        if df == 0 or not np.isfinite(df):
            if verbose:
                print(f"Iteration {i}: df is non-finite or zero at x = {x}")
            break

        dx = -f / df

        if not np.isfinite(dx):
            if verbose:
                print(f"Iteration {i}: dx is non-finite at x = {x}, f = {f}, df = {df}")
            break
        x += dx
        if abs(dx) < tol:
            return x
    return x

# --- Chi Solver ---
def Chi_newton(mu, alpha, r0, vr, dt, tol=1e-8, max_iter=100, verbose=False):
    chi0 = np.sqrt(mu) * dt * abs(alpha)
    r = norm(r0)

    def F(chi):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        return (r * vr / np.sqrt(mu)) * chi**2 * C + (1 - r * alpha) * chi**3 * S + r * chi - np.sqrt(mu) * dt
    
    def dF(chi):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        return (r * vr / np.sqrt(mu)) * chi * (1 - z * S) + (1 - r * alpha) * chi**2 * C + r
    
    if verbose:
        print(f"alpha={alpha}, chi0={chi0}, z={alpha * chi0**2}")

    if not np.isfinite(chi0):
        raise ValueError("Inital guess chi0 is not finite")
    
    if np.abs(alpha * chi0**2) > 1e6:
        raise ValueError("Chi0 guess leads to extreme z value")

    return Newton_Solver(tol, F, dF, chi0, max_iter=max_iter, verbose=verbose)

# --- Orbit Propagation ---
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

    f = 1 - (chi**2 / r0) * C
    g = dt - (1 / np.sqrt(mu)) * chi**3 * S

    r_vec = f * r0_vec + g * v0_vec
    r = norm(r_vec)

    fdot = (np.sqrt(mu) / (r * r0)) * chi * (z * S - 1)
    gdot = 1 - (chi**2 / r) * C

    v_vec = fdot * r0_vec + gdot * v0_vec
    
    return r_vec, v_vec

def plot_mean_true(e_list,r_p,mu,dt,time):
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

            for i in tqdm(range(steps), desc=f"Elliptical e={e0:.2g}"):
                try:
                    r_vec, v_vec = propagate(r_vec, v_vec, mu, dt)
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

            plt.scatter(true_anomalies,mean_anomalies, s=1, label=f"e={e0:.2g}")
            plt.title("Elliptical Orbits")
            plt.xlabel("True Anomaly (deg)")
            plt.ylabel("Mean Anomaly (deg)")
            plt.grid(True)

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

            for _ in tqdm(range(steps), desc=f"Elliptical e={e0:.2g}"):
                try:
                    r_vec_f, v_vec_f = propagate(r_vec_f, v_vec_f, mu, dt)
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
            
            for _ in tqdm(range(steps), desc=f"Elliptical e={e0:.2g}"):
                try:
                    r_vec_b, v_vec_b = propagate(r_vec_b, v_vec_b, mu, -dt)
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

            plt.scatter(true_anomalies,mean_anomalies, s=1, label=f"e={e0:.2g}")
            plt.xlabel("True Anomaly (deg)")
            plt.ylabel("Mean Anomaly (rad)")
            plt.grid(True)
            plt.ylim([-np.pi,np.pi])
            
    plt.legend()

# --- Main Plotting Routine ---
if __name__ == "__main__":
    start_time = time.time()

    dt = 1  # time step in seconds
    e_list_elpic = [1e-8, 0.5, 0.9]
    e_list_para = [1]
    e_list_hyper = [1.2, 1.5, 2.0, 5, 12]
    r_p = 10000  # km
    mu = 398600.4418  # km^3/s^2
    time_total = 5400  # seconds

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_mean_true(e_list_elpic, r_p, mu, dt, time_total)

    plt.subplot(1, 3, 2)
    plot_mean_true(e_list_para, r_p, mu, dt, time_total)
    plt.title("Parabolic Orbit")

    plt.subplot(1, 3, 3)
    plot_mean_true(e_list_hyper, r_p, mu, dt, time_total)
    plt.title("Hyperbolic Orbits")

    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")